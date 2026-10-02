const state = {
  token: sessionStorage.getItem("agent_api_token") || "",
  sessionId: sessionStorage.getItem("agent_session_id") || "",
  images: [],
  objectUrls: [],
};

const byId = (id) => document.getElementById(id);
const tokenInput = byId("api-token");
const questionInput = byId("question");
const messageList = byId("message-list");
const imageInput = byId("image-input");
const uploadPreview = byId("upload-preview");
const sendButton = byId("send-button");

function requestHeaders(json = false) {
  const result = {};
  if (json) result["Content-Type"] = "application/json";
  if (state.token) result.Authorization = `Bearer ${state.token}`;
  return result;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...requestHeaders(Boolean(options.body)),
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    let detail = `请求失败 (${response.status})`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
      if (Array.isArray(detail)) {
        detail = detail.map((item) => item.msg).join("；");
      }
    } catch (_) {
      // Preserve the HTTP fallback when the response is not JSON.
    }
    throw new Error(detail);
  }
  return response;
}

function showToast(message) {
  const toast = byId("toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 2600);
}

function setServiceStatus(id, ok, label) {
  const target = byId(id);
  target.className = `status-dot ${ok ? "ok" : "error"}`;
  target.textContent = label;
}

async function checkStatus() {
  try {
    await api("/v1/health");
    setServiceStatus("health-status", true, "在线");
  } catch (_) {
    setServiceStatus("health-status", false, "离线");
  }
  try {
    await api("/v1/ready");
    setServiceStatus("ready-status", true, "就绪");
  } catch (_) {
    setServiceStatus("ready-status", false, "未就绪");
  }
}

function ensureSession() {
  if (!state.sessionId) {
    state.sessionId = `web_${crypto.randomUUID().replaceAll("-", "")}`;
    sessionStorage.setItem("agent_session_id", state.sessionId);
  }
  byId("session-short").textContent = state.sessionId.slice(-12);
  return state.sessionId;
}

function appendMessage(role, text, data = null) {
  byId("empty-state")?.remove();
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const head = document.createElement("div");
  head.className = "message-head";
  const author = document.createElement("strong");
  author.textContent = role === "user" ? "用户" : "客服 Agent";
  const time = document.createElement("span");
  time.textContent = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
  head.append(author, time);

  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;
  article.append(head, body);

  if (data) {
    const metadata = document.createElement("div");
    metadata.className = "message-meta";
    addBadge(metadata, data.agent.intent);
    addBadge(metadata, `${data.metrics.total_latency_ms} ms`);
    if (data.agent.handoff_required) addBadge(metadata, "已转人工", true);
    article.append(metadata);
    if (data.images?.length) renderAnswerImages(article, data.images);
    renderFeedback(article, data);
  }

  messageList.append(article);
  messageList.scrollTop = messageList.scrollHeight;
}

function addBadge(container, text, handoff = false) {
  const badge = document.createElement("span");
  badge.className = `badge${handoff ? " handoff" : ""}`;
  badge.textContent = text;
  container.append(badge);
}

async function renderAnswerImages(container, images) {
  const gallery = document.createElement("div");
  gallery.className = "answer-images";
  container.append(gallery);
  for (const image of images) {
    try {
      const response = await api(image.url);
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      state.objectUrls.push(objectUrl);
      const figure = document.createElement("figure");
      const element = document.createElement("img");
      element.alt = `知识库操作示意图 ${image.image_id}`;
      element.src = objectUrl;
      const caption = document.createElement("figcaption");
      caption.textContent = image.image_id;
      figure.append(element, caption);
      gallery.append(figure);
    } catch (_) {
      // Optional image assets must not hide the text answer.
    }
  }
}

function renderFeedback(container, data) {
  const row = document.createElement("div");
  row.className = "feedback-row";
  row.append(document.createTextNode("本次回答："));
  for (let rating = 1; rating <= 5; rating += 1) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "rating-button";
    button.textContent = String(rating);
    button.title = `${rating} 分`;
    button.addEventListener("click", async () => {
      try {
        await api("/v1/feedback", {
          method: "POST",
          body: JSON.stringify({
            session_id: data.session_id,
            request_id: data.request_id,
            rating,
            comment: "web_console_rating",
          }),
        });
        row.querySelectorAll("button").forEach((item) => item.classList.remove("selected"));
        button.classList.add("selected");
        showToast("评分已记录");
      } catch (error) {
        showToast(error.message);
      }
    });
    row.append(button);
  }
  container.append(row);
}

function renderInspector(data) {
  byId("trace-id").textContent = data.request_id;
  const summary = byId("agent-summary");
  summary.className = "inspector-section";
  summary.replaceChildren();
  const grid = document.createElement("div");
  grid.className = "agent-grid";
  [
    ["识别意图", data.agent.intent],
    ["执行动作", data.agent.action],
    ["决策置信度", `${Math.round(data.agent.confidence * 100)}%`],
    ["人工工单", data.agent.ticket_id || "不需要"],
  ].forEach(([label, value]) => {
    const item = document.createElement("div");
    const name = document.createElement("span");
    name.textContent = label;
    const result = document.createElement("strong");
    result.textContent = value;
    item.append(name, result);
    grid.append(item);
  });
  summary.append(grid);

  byId("metric-list").innerHTML = `
    <div><dt>总耗时</dt><dd>${data.metrics.total_latency_ms} ms</dd></div>
    <div><dt>检索</dt><dd>${data.metrics.retrieval_latency_ms} ms</dd></div>
    <div><dt>模型</dt><dd>${data.metrics.llm_latency_ms} ms</dd></div>
    <div><dt>模型尝试</dt><dd>${data.metrics.llm_attempt_count}</dd></div>`;

  const references = byId("reference-list");
  references.className = "reference-list";
  references.replaceChildren();
  if (!data.references.length) {
    references.className = "reference-list empty-copy";
    references.textContent = "本次请求未使用知识资料";
    return;
  }
  data.references.slice(0, 8).forEach((reference) => {
    const item = document.createElement("div");
    item.className = "reference-item";
    const title = document.createElement("strong");
    title.textContent = `${reference.product} · ${reference.chunk_id}`;
    const detail = document.createElement("span");
    detail.textContent = `${reference.retrieval_type} / ${reference.score.toFixed(3)}`;
    item.append(title, detail);
    references.append(item);
  });
}

async function submitQuestion(event) {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  appendMessage("user", question);
  questionInput.value = "";
  sendButton.disabled = true;
  byId("request-state").textContent = "Agent 处理中";
  try {
    const response = await api("/v1/chat", {
      method: "POST",
      body: JSON.stringify({
        question,
        session_id: ensureSession(),
        images: state.images.map((item) => item.dataUrl),
      }),
    });
    const payload = await response.json();
    appendMessage("assistant", payload.data.answer, payload.data);
    renderInspector(payload.data);
    clearUploads();
    if (payload.data.agent.ticket_id) {
      showToast(`已创建工单 ${payload.data.agent.ticket_id}`);
    }
  } catch (error) {
    appendMessage("assistant", `请求未完成：${error.message}`);
  } finally {
    sendButton.disabled = false;
    byId("request-state").textContent = "空闲";
    questionInput.focus();
  }
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function handleImages() {
  const files = [...imageInput.files].slice(0, 3 - state.images.length);
  for (const file of files) {
    if (file.size > 4 * 1024 * 1024) {
      showToast(`${file.name} 超过单图 4 MB 限制`);
      continue;
    }
    state.images.push({ file, dataUrl: await fileToDataUrl(file) });
  }
  imageInput.value = "";
  renderUploadPreview();
}

function renderUploadPreview() {
  uploadPreview.replaceChildren();
  uploadPreview.hidden = state.images.length === 0;
  state.images.forEach((item, index) => {
    const wrapper = document.createElement("div");
    wrapper.className = "preview-item";
    const image = document.createElement("img");
    image.alt = `待上传图片 ${index + 1}`;
    image.src = item.dataUrl;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", "移除图片");
    remove.addEventListener("click", () => {
      state.images.splice(index, 1);
      renderUploadPreview();
    });
    wrapper.append(image, remove);
    uploadPreview.append(wrapper);
  });
  byId("input-hint").textContent = state.images.length
    ? `已选择 ${state.images.length}/3 张`
    : "最多 3 张图片";
}

function clearUploads() {
  state.images = [];
  renderUploadPreview();
}

async function startNewSession() {
  if (state.sessionId) {
    try {
      await api(`/v1/sessions/${state.sessionId}`, { method: "DELETE" });
    } catch (_) {
      // The local UI reset remains useful when the server session expired.
    }
  }
  state.objectUrls.forEach((url) => URL.revokeObjectURL(url));
  state.objectUrls = [];
  state.sessionId = "";
  sessionStorage.removeItem("agent_session_id");
  messageList.innerHTML = '<div class="empty-state" id="empty-state"><strong>新会话已创建</strong><p>输入新的问题开始处理。</p></div>';
  byId("session-short").textContent = "未创建";
  byId("trace-id").textContent = "等待请求";
  showToast("会话已清空");
}

const statusLabels = {
  open: "待处理",
  in_progress: "处理中",
  resolved: "已解决",
  closed: "已关闭",
};
const priorityLabels = { normal: "普通", high: "高", urgent: "紧急" };

async function loadOperationsSummary() {
  try {
    const response = await api("/v1/operations/summary");
    const summary = await response.json();
    byId("active-ticket-count").textContent = summary.active_ticket_count;
    byId("urgent-ticket-count").textContent = summary.urgent_open_count;
    byId("feedback-count").textContent = summary.feedback_count;
    byId("feedback-average").textContent = summary.feedback_average_rating === null
      ? "暂无"
      : `${summary.feedback_average_rating.toFixed(2)} / 5`;
  } catch (error) {
    showToast(`运营摘要加载失败：${error.message}`);
  }
}

async function loadTickets() {
  const status = byId("ticket-filter").value;
  const body = byId("ticket-body");
  body.innerHTML = '<tr><td colspan="6" class="table-empty">加载中</td></tr>';
  try {
    const response = await api(
      `/v1/tickets?limit=100${status ? `&status=${status}` : ""}`,
    );
    const tickets = await response.json();
    byId("ticket-count").textContent = `${tickets.length} 个工单`;
    body.replaceChildren();
    if (!tickets.length) {
      body.innerHTML = '<tr><td colspan="6" class="table-empty">当前筛选条件下没有工单</td></tr>';
      return;
    }
    tickets.forEach((ticket) => body.append(buildTicketRow(ticket)));
  } catch (error) {
    body.replaceChildren();
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.className = "table-empty";
    cell.textContent = error.message;
    row.append(cell);
    body.append(row);
  }
}

function buildTicketRow(ticket) {
  const row = document.createElement("tr");
  const idCell = document.createElement("td");
  const code = document.createElement("code");
  code.textContent = ticket.ticket_id;
  idCell.append(code);

  const questionCell = document.createElement("td");
  questionCell.className = "question-cell";
  questionCell.title = ticket.question;
  questionCell.textContent = ticket.question;

  const priorityCell = document.createElement("td");
  priorityCell.className = `priority-${ticket.priority}`;
  priorityCell.textContent = priorityLabels[ticket.priority];

  const statusCell = document.createElement("td");
  statusCell.textContent = statusLabels[ticket.status];
  const timeCell = document.createElement("td");
  timeCell.textContent = formatTimestamp(ticket.updated_at);

  const actionCell = document.createElement("td");
  const action = document.createElement("button");
  action.type = "button";
  action.className = "ticket-action";
  action.textContent = ticket.status === "closed" ? "已完成" : "推进状态";
  action.disabled = ticket.status === "closed";
  action.addEventListener("click", () => advanceTicket(ticket));
  actionCell.append(action);

  row.append(idCell, questionCell, priorityCell, statusCell, timeCell, actionCell);
  return row;
}

function formatTimestamp(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  const milliseconds = numeric > 1_000_000_000_000
    ? numeric / 1_000_000
    : numeric * 1000;
  return new Date(milliseconds).toLocaleString();
}

async function advanceTicket(ticket) {
  const nextStatus = {
    open: "in_progress",
    in_progress: "resolved",
    resolved: "closed",
    closed: "closed",
  }[ticket.status];
  if (nextStatus === ticket.status) return;
  try {
    await api(`/v1/tickets/${ticket.ticket_id}`, {
      method: "PATCH",
      body: JSON.stringify({ status: nextStatus }),
    });
    showToast(`工单已更新为${statusLabels[nextStatus]}`);
    await Promise.all([loadTickets(), loadOperationsSummary()]);
  } catch (error) {
    showToast(error.message);
  }
}

function switchView(viewName) {
  const target = document.querySelector(`.nav-item[data-view="${viewName}"]`);
  if (!target) return;
  target.click();
}

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
    document.querySelectorAll(".view").forEach((view) => {
      view.classList.toggle("active", view.id === `${button.dataset.view}-view`);
    });
    if (button.dataset.view === "tickets") {
      loadTickets();
      loadOperationsSummary();
    }
  });
});

// 深链接：支持 ?view=tickets 直达工单队列，便于分享与自动化截图
const requestedView = new URLSearchParams(window.location.search).get("view");
if (requestedView) {
  switchView(requestedView);
}

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    questionInput.focus();
  });
});

tokenInput.value = state.token;
byId("toggle-token").addEventListener("click", () => {
  tokenInput.type = tokenInput.type === "password" ? "text" : "password";
  byId("toggle-token").textContent = tokenInput.type === "password" ? "显示" : "隐藏";
});
byId("save-token").addEventListener("click", () => {
  state.token = tokenInput.value.trim();
  if (state.token) sessionStorage.setItem("agent_api_token", state.token);
  else sessionStorage.removeItem("agent_api_token");
  showToast("连接设置已应用");
  checkStatus();
});

byId("chat-form").addEventListener("submit", submitQuestion);
imageInput.addEventListener("change", handleImages);
byId("new-session").addEventListener("click", startNewSession);
byId("refresh-tickets").addEventListener("click", () => {
  loadTickets();
  loadOperationsSummary();
});
byId("ticket-filter").addEventListener("change", loadTickets);

ensureSession();
checkStatus();
