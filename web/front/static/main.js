'use strict';
import { addOrUpdateChat, getChats, getActiveChatId, setActiveChatId, removeChat } from './history.js'

/** --------------------------- Constants --------------------------- */
const API_BASE = `${window.location.origin}/api/`;

/** --------------------------- Global state --------------------------- */
const state = {
    filters: {},
    selected_documents: []
};
let categories = [];
let searchPaths = [];

/** --------------------------- Init --------------------------- */
$(function init() {
    bindUI();

    renderChatList();
    let chatId = getActiveChatId();
    if (chatId) {
        activate(chatId);
    } else {
        handleNewChat();
    }

    loadMetaData();
});

// ---------------------- Meta‑data flow ---------------------- //

async function loadMetaData() {
    [categories, searchPaths] = await Promise.all([
        fetchJSON('meta_data/categories'),
        fetchJSON('meta_data/search_paths'),
    ]);

    renderCategorySelectors();
    renderSearchPathDatalist();
}

function renderCategorySelectors() {
    const $filter_container = $('#filter-categories').empty();
    categories.forEach(({ id, name, values }) => {
        $filter_container.append(`
      <label>${name}</label>
      <select multiple class="form-control" id="filter-category-${id}">
      <option value="">Select none</option>
      ${values.map(v => `<option value="${v}">${v}</option>`)}
      </select>
    `);
    });

}

function renderSearchPathDatalist() {
    const $list = $('#searchPathList').empty();
    searchPaths.forEach(path => $list.append(`<option value="${path.folder}" />`));
}


/** --------------------------- UI Bindings --------------------------- */
function bindUI() {
    $('#new-chat-btn').on('click', handleNewChat);

    // Send on Enter (Shift+Enter makes a newline).
    $('#message').on('keypress', function onKeypress(e) {
        if (e.which !== 13 || e.shiftKey) return;

        e.preventDefault();
        const $input = $(this);
        const text = $input.val().trim();
        if (!text) return;

        $input.prop('disabled', true).val('');
        $('#output .thinking').remove();

        const $output = $('#output')
        $output.append(`<div class="user">${escapeHtml(text)}</div>`);
        $output.append(`<div class="thinking tool"></div>`);
        const $thinking = $('#output .thinking')
        updateDivText($thinking, "Thinking");
        $output.scrollTop($output[0].scrollHeight);
        sendPrompt(text, $input);
    });

    // Delegate chat item clicks to the list container.
    $('#chats').on('click', '.chat-item', function () {
        const chatId = $(this).data('id');
        activate(chatId);
    });

    $('#clear-files-btn').on('click', clearSelectedFiles);

    /* Filter controls */
    $('#applyFiltersBtn').on('click', applyFilters);
    $('#resetFiltersBtn').on('click', resetFilters);
}

/** --------------------------- Chat List --------------------------- */
function handleNewChat() {
    $.get(`${API_BASE}chat/get_new_chat_id`)
        .done((data) => {
            const chatId = data.chat_id
            addOrUpdateChat(chatId);
            renderChatList();
            activate(chatId);
        })
        .fail((xhr) => {
            console.error('Failed to create a new chat:', xhr?.responseText || xhr);
        });
}

function renderChatList() {
    const items = getChats().map((id, i) => {
        return `
<li class="chat-item list-group-item d-flex align-items-center" data-id="${id}">
  <i class="bi bi-chat-left-text me-2"></i>
  Chat #${i + 1}
  <i class="bi bi-trash ms-auto text-danger deleteChat" data-id="${id}"></i>
</li>`;
    })
        .join('');

    $('#chats').html(items);

    $('.deleteChat').click(function () {
        const chatId = $(this).data('id');
        deleteChat(chatId)
    })
}

function deleteChat(chatId) {
    if (!chatId) return;
    if (!confirm("Are you sure?")) return;

    $.ajax({
        url: `${API_BASE}chat/${chatId}`,
        type: "DELETE",
    })
        .done(() => {
            removeChat(chatId);
            renderChatList();
            renderChat([]);
        })
        .fail((xhr) => {
            console.error("Failed to delete chat:", xhr?.responseText || xhr);
        });
}
/** --------------------------- Chat Activation --------------------------- */
function activate(chatId) {
    setActiveChatId(chatId);
    $('.chat-item').removeClass('active');
    $(`#chats .chat-item[data-id="${chatId}"]`).addClass('active');
    $.get(`${API_BASE}chat/${chatId}`)
        .done((data) => {
            renderChat(data);
            // Check for pending interrupts (e.g. after page refresh)
            checkPendingInterrupt(chatId);
        })
        .fail((xhr) => {
            console.error('Failed to load chat:', xhr?.responseText || xhr);
            renderChat([]);
        });
}

function renderChat(data) {
    const messages = data.reverse()
    const $output = $('#output');
    $output.empty();

    messages?.forEach((msg) => {
        const human = msg?.last_conversation?.request;
        const ai = msg?.last_conversation?.response;
        if (human) {
            $('<div>', { class: 'user', text: human.content }).appendTo($output);
        }
        if (ai) {
            $('<div>', { class: 'container system', text: ai.content }).appendTo($output);
        }
        addDocs(msg?.last_conversation?.documents)
    });

    $output.scrollTop($output[0].scrollHeight);

    clearSelectedFiles();
}

/** --------------------------- Messaging & Streaming --------------------------- */
function sendPrompt(inputValue, $input) {
    const chatId = getActiveChatId();
    if (!chatId) {
        console.warn('No active chat.');
        $input.prop('disabled', false);
        return;
    }

    fetch(`${API_BASE}chat/${chatId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query: inputValue,
            selected_documents: state.selected_documents,
            filter: state.filters,
        }),
    })
        .then(() => openSSE(chatId, $input))
        .catch((err) => {
            console.error('Request error:', err);
            $input.prop('disabled', false);
        });
}

function openSSE(chatId, $input) {
    const es = new EventSource(`${API_BASE}chat/${chatId}/stream`);
    const $output = $('#output');
    const $tool = $('<div>').addClass('container tool').appendTo($output);
    const $msg = $('<div>').addClass('container system').appendTo($output);

    es.onmessage = (e) => {
        const event = JSON.parse(e.data)

        // Handle interrupt events (human-in-the-loop)
        if (event.type == 'interrupt') {
            es.close();
            $tool.remove();
            $msg.remove();
            $('#output .thinking').remove();
            const interruptData = event.interrupts[0].value;
            handleInterrupt(chatId, interruptData, $input);
            return;
        }

        if (event.type == 'AIMessageChunk') {
            updateDivText($tool, "");
            appendToken($msg, event.content);
        }

        if (event.type == 'tool') {
            $('#output .thinking').remove();
            updateDivText($tool, event.content);
        }
        $output.scrollTop($output[0].scrollHeight);
    };

    es.addEventListener('end', () => {
        es.close();
        fetchPostData(chatId);
        $input.prop('disabled', false);
    });

    es.addEventListener('error', (e) => {
        console.error('SSE error:', e);
        es.close();
        $input.prop('disabled', false);
        $('#output .thinking').remove();
        updateDivText($tool,'Oops, I have an error, pleae try asking something else');
    });

}

function fetchPostData(chatId) {
    fetch(`${API_BASE}chat/${chatId}/current_state`)
        .then((res) => {
            if (res.ok) {
                res.json().then(data => {
                    const docs = data?.last_conversation?.documents;
                    if (docs && docs.length > 0) {
                        addDocs(docs);
                    }
                }).catch(err => {
                    console.log(err)
                })
            }
        })
        .catch((err) => console.error('current_state error:', err));
}

/** --------------------------- File Selection --------------------------- */
function check_file(cb) {
    debugger
    const val = cb?.value;
    if (!val) return;

    if (cb.checked) {
        if (!state.selected_documents.includes(val)) state.selected_documents.push(val);
    } else {
        state.selected_documents = state.selected_documents.filter((x) => x !== val);
    }
    renderFileControls();
}

function clearSelectedFiles() {
    state.selected_documents = [];
    $('.select-file-chk').prop('checked', false);
    renderFileControls();
}

function renderFileControls() {
    const hasFiles = state.selected_documents.length > 0;
    $('#files-warning').toggle(hasFiles);
    $('#clear-files-btn').toggle(hasFiles);
}


/* --------------------------- Filters --------------------------- */
function applyFilters() {
    state.filters['category_ids'] = []
    categories.forEach(ct => {
        const vals = $(`#filter-category-${ct.id}`).val()?.filter(v => v);
        if (vals) {
            state.filters['category_ids'].push({
                'id': ct.id,
                'categories': vals
            })
        }
    })
    state.filters.file = $('#filter-file').val().trim();
    state.filters.folder = $('#filter-search-path').val().trim();
    state.filters.created_from = $('#filter-created-date-from').val();
    state.filters.created_to = $('#filter-created-date-to').val();
    state.filters.updated_from = $('#filter-updated-date-from').val();
    state.filters.updated_to = $('#filter-updated-date-to').val();
    state.filters.author = $('#filter-file-author').val().trim();
    state.page = 1;
    bootstrap.Modal.getInstance('#filterModal').hide();
}

function resetFilters() {
    state.filters = {};
    $('#filterForm').trigger('reset');
    bootstrap.Modal.getInstance('#filterModal').hide();
}

/** --------------------------- Interrupt Handling (Human-in-the-Loop) --------------------------- */

function checkPendingInterrupt(chatId) {
    fetch(`${API_BASE}chat/${chatId}/interrupt_status`)
        .then(res => res.json())
        .then(data => {
            if (data.has_interrupt && data.interrupt_data) {
                const interruptData = data.interrupt_data[0].value;
                const $input = $('#message');
                $input.prop('disabled', true);
                handleInterrupt(chatId, interruptData, $input);
            }
        })
        .catch(err => console.error('interrupt_status error:', err));
}

function handleInterrupt(chatId, data, $input) {
    if (data.type === 'email_input_request') {
        renderEmailInputForm(chatId, data, $input);
    } else if (data.type === 'email_confirmation_request') {
        renderEmailConfirmation(chatId, data, $input);
    }
}

function renderEmailInputForm(chatId, data, $input) {
    const $output = $('#output');
    const fields = data.fields || [];
    const emailField = fields.find(f => f.name === 'email') || { name: 'email', label: 'Email address' };
    const nameField = fields.find(f => f.name === 'name') || { name: 'name', label: 'Recipient name' };

    const $form = $(`
        <div class="container system email-interrupt-form">
            <h6>${escapeHtml(data.message)}</h6>
            <div class="mb-2">
                <label class="form-label">${escapeHtml(nameField.label)}</label>
                <input type="text" class="form-control form-control-sm" id="interrupt-name"
                       placeholder="Recipient name" />
            </div>
            <div class="mb-2">
                <label class="form-label">${escapeHtml(emailField.label)}</label>
                <input type="email" class="form-control form-control-sm" id="interrupt-email"
                       placeholder="email@example.com" />
            </div>
            <div class="d-flex gap-2 mt-2">
                <button class="btn btn-primary btn-sm" id="interrupt-submit-btn">Submit</button>
                <button class="btn btn-outline-secondary btn-sm" id="interrupt-cancel-btn">Cancel</button>
            </div>
        </div>
    `);

    $output.append($form);
    $output.scrollTop($output[0].scrollHeight);

    $form.find('#interrupt-submit-btn').on('click', () => {
        const email = $form.find('#interrupt-email').val().trim();
        const name = $form.find('#interrupt-name').val().trim();
        $form.find('button').prop('disabled', true);
        resumeGraph(chatId, { email, name }, $input);
    });

    $form.find('#interrupt-cancel-btn').on('click', () => {
        $form.find('button').prop('disabled', true);
        resumeGraph(chatId, { email: '', name: '' }, $input);
    });
}

function renderEmailConfirmation(chatId, data, $input) {
    const $output = $('#output');
    const preview = data.preview || {};

    const $confirm = $(`
        <div class="container system email-confirm-form">
            <h6>${escapeHtml(data.message)}</h6>
            <div class="card card-body mb-2">
                <p class="mb-1"><strong>To:</strong> ${escapeHtml(preview.to_name || '')} &lt;${escapeHtml(preview.to_email || '')}&gt;</p>
                <p class="mb-1"><strong>Subject:</strong> ${escapeHtml(preview.subject || '')}</p>
                <p class="mb-0"><strong>Preview:</strong> ${escapeHtml(preview.body_preview || '')}</p>
            </div>
            <div class="d-flex gap-2 mt-2">
                <button class="btn btn-success btn-sm" id="confirm-send-btn">Send</button>
                <button class="btn btn-outline-secondary btn-sm" id="confirm-cancel-btn">Cancel</button>
            </div>
        </div>
    `);

    $output.append($confirm);
    $output.scrollTop($output[0].scrollHeight);

    $confirm.find('#confirm-send-btn').on('click', () => {
        $confirm.find('button').prop('disabled', true);
        resumeGraph(chatId, { confirmed: true }, $input);
    });

    $confirm.find('#confirm-cancel-btn').on('click', () => {
        $confirm.find('button').prop('disabled', true);
        resumeGraph(chatId, { confirmed: false }, $input);
    });
}

function resumeGraph(chatId, value, $input) {
    const $output = $('#output');
    $output.append(`<div class="thinking tool"></div>`);
    const $thinking = $('#output .thinking');
    updateDivText($thinking, "Processing...");

    fetch(`${API_BASE}chat/${chatId}/resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(value),
    })
        .then(() => openSSE(chatId, $input))
        .catch((err) => {
            console.error('Resume error:', err);
            $input.prop('disabled', false);
            $('#output .thinking').remove();
        });
}

/* --------------------------- Helpers --------------------------- */
async function fetchJSON(relativePath) {
    const res = await fetch(`${API_BASE}${relativePath}`);
    if (!res.ok) throw res;
    return res.json();
}

function appendToken($target, chunk) {
    if (!chunk) return;
    $target.append(document.createTextNode(chunk))
}

function escapeHtml(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


function updateDivText($div, newText) {
    $div.fadeOut(500, function () {
        $div.text(newText).fadeIn(500);
    });
}

function addDocs(docs) {
    if (!docs || docs.length == 0) return;
    const grouped_docs = []
    docs.forEach(doc => {
        let gd = grouped_docs.find(d => d.file_id == doc.metadata.file_id);
        if (!gd) {
            gd = { "file_id": doc.metadata.file_id, "original_file_name": doc.metadata.original_file_name, "folder": doc.metadata.folder, "contents": [] }
            grouped_docs.push(gd)
        }
        gd.contents.push(doc.page_content)
    })
    const $output = $('#output')
    let doc_list = `<div class="documents"><h5>Documents</h5>`;
    grouped_docs.forEach(d => {
        let li = `<label class="select-card w-100">
    <input type="checkbox" class="d-none select-file-chk" value="${d.file_id}" />
    <div class="card shadow-sm rounded-3 position-relative">
      <span class="check-badge badge rounded-pill text-bg-primary">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor"
             class="bi bi-check2" viewBox="0 0 16 16">
          <path d="M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0l-3-3a.5.5 0 1 1 .708-.708L6.5 10.293l6.646-6.647a.5.5 0 0 1 .708 0"/>
        </svg>
        Selected
      </span>
      <div class="card-body">
        <a target="_blank" href="${API_BASE}document_manager/download/${d.file_id}">${d.original_file_name}</a>
        <h4>${d.folder}</h4>
        <div class="vstack gap-3">
            ${d.contents.map(c => `<div class='line-clamp-2'>${escapeHtml(c)}</div>`).join('')}
        </div>
      </div>
    </div>
  </label>`
        doc_list += li;
    });
    doc_list += `</div>`;
    $output.append(doc_list)
    $('.select-file-chk').unbind('change');
    $('.select-file-chk').change(function () {
        check_file($(this)[0])
    });
}