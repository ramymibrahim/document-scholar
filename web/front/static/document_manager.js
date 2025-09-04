'use strict';
const API_BASE = `${window.location.origin}/api/`;

/* --------------------------- Global state --------------------------- */
const state = {
  page: 1,
  size: 10,
  total: 0,
  sortField: null,
  sortDir: 'asc',
  rows: [],
  filters: {}
};
let categories = [];
let searchPaths = [];

/* ------------------------------ Init ------------------------------- */
$(async () => {

  /* Page‑size selector */
  $('#page-size').on('change', function () {
    state.size = +this.value;
    state.page = 1;
    loadPage();
  });

  /* Header click for server‑side sorting */
  $(document).on('click', 'th.sortable', function () {
    const field = $(this).data('field');
    if (!field) return;

    if (state.sortField === field) {
      state.sortDir = state.sortDir === 'asc' ? 'desc' : 'asc';
    } else {
      state.sortField = field;
      state.sortDir = 'asc';
    }

    state.page = 1;           // reset to first page on new sort
    loadPage();
  });

  /* Pagination click */
  $(document).on('click', '#pagination li.page-item:not(.disabled) a.page-link', function (e) {
    e.preventDefault();
    const target = +$(this).data('page');
    if (target && target !== state.page) {
      state.page = target;
      loadPage();
    }
  });

  /* Filter controls */
  $('#applyFiltersBtn').on('click', applyFilters);
  $('#resetFiltersBtn').on('click', resetFilters);

  /* Crud controls */
  $('#saveBtn').on('click', handleUpload);


  /* Initial load */
  await loadPage();
  await loadMetaData();
});

/* ------------------------ Data fetch & render ---------------------- */
async function loadPage() {
  const params = { ...state.filters };
  params['page'] = state.page;
  params['size'] = state.size;

  // Sorting
  if (state.sortField) {
    params['sort_field'] = state.sortField;
    params['sort_dir'] = state.sortDir;
  }

  const { items, total } = await fetchJSON(`document_manager`, {
    method: 'POST', body: JSON.stringify(params), "Content-Type": "application/json",
  });
  state.rows = items;
  state.total = total ?? items.length;

  renderTable();
  renderPagination();
  updateSortIcons();
}

/* ---------------------------- Table ----------------------------- */
function renderTable() {
  const $tbody = $('#data-table').empty();

  state.rows.forEach((r, i) => {
    $tbody.append(`
      <tr>
        <th scope="row">${(state.page - 1) * state.size + i + 1}</th>
        <td>${r.folder ?? ""}</td>
        <td>${r.original_file_name}</td>
        <td>${r.created_at ?? ""}</td>
        <td>${r.updated_at ?? ""}</td>
        <td>${r.author ?? ""}</td>
        <td>${get_category_vals(r)}</td>
        <td class="text-end">
          <button class="btn btn-sm btn-outline-primary" title="View details" onclick="view('${r.id}')">
            <i class="bi bi-eye"></i>
          </button>
           </td>
          <td class="text-end">
          <a class="btn btn-sm btn-outline-primary" href="${API_BASE}document_manager/download/${r.id}" title="Download">
            <i class="bi bi-download"></i>
          </a>
           </td>
        <td class="text-end">
             <button class="btn btn-sm btn-outline-danger" title="Delete" onclick="delete_file('${r.id}')">
            <i class="bi bi-trash"></i>
          </button>
        </td>
      </tr>`);
  });
}

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
  const $container = $('#categories').empty();
  categories.forEach(({ id, name, values }) => {
    $container.append(`
      <label>${name}</label>
      <select class="form-control" id="category-${id}">
        <option value="">Select ${name}</option>
        ${values.map(v => `<option value="${v}">${v}</option>`)}
      </select>
    `);
  });


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


/* ------------------------- Pagination -------------------------- */
function renderPagination() {
  const pages = Math.max(1, Math.ceil(state.total / state.size));
  const $p = $('#pagination').empty();

  const item = (label, pg, disabled = false, active = false) =>
    `<li class="page-item ${disabled ? 'disabled' : ''} ${active ? 'active' : ''}">
       <a class="page-link" href="#" data-page="${pg}">${label}</a>
     </li>`;

  $p.append(item('«', state.page - 1, state.page === 1));

  const window = 2;
  const start = Math.max(1, state.page - window);
  const end = Math.min(pages, state.page + window);

  if (start > 1) $p.append(item('1', 1));
  if (start > 2) $p.append('<li class="page-item disabled"><span class="page-link">…</span></li>');

  for (let pg = start; pg <= end; pg++) {
    $p.append(item(pg, pg, false, pg === state.page));
  }

  if (end < pages - 1) $p.append('<li class="page-item disabled"><span class="page-link">…</span></li>');
  if (end < pages) $p.append(item(pages, pages));

  $p.append(item('»', state.page + 1, state.page === pages));
}

/* ------------------------- Sorting UI -------------------------- */
function updateSortIcons() {
  $('th.sortable .sort-icon')
    .removeClass('bi-sort-up bi-sort-down')
    .addClass('bi-arrow-down-up');

  if (!state.sortField) return;
  const $icon = $(`th.sortable[data-field="${state.sortField}"] .sort-icon`);
  $icon
    .removeClass('bi-arrow-down-up')
    .addClass(state.sortDir === 'asc' ? 'bi-sort-up' : 'bi-sort-down');
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
  loadPage();
  bootstrap.Modal.getInstance('#filterModal').hide();
}

function resetFilters() {
  state.filters = {};
  $('#filterForm').trigger('reset');
  state.page = 1;
  loadPage();
  bootstrap.Modal.getInstance('#filterModal').hide();
}

/* --------------------------- Helpers --------------------------- */
async function fetchJSON(relativePath, req = null) {
  const res = await fetch(`${API_BASE}${relativePath}`, req);
  if (!res.ok) throw res;
  return res.json();
}

/* -------------------------- CRUD ------------------------------ */
async function handleUpload() {
  const $status = $('#uploadStatus').text('').removeClass('text-danger');
  const file = $('#fileInput').prop('files')[0];

  if (!file) {
    return $status.text('Please select a file.').addClass('text-danger mt-2');
  }

  const formData = buildFormData(file);

  try {
    toggleButtons(false);
    await fetch(`${API_BASE}document_manager/upload`, { method: 'POST', body: formData });

    showNotification('Upload successful!', 'success');
    bootstrap.Modal.getInstance('#uploadModal').hide();
    clearFields();
    loadPage();
  } catch (err) {
    console.log(err)
    let msg = 'Upload failed.';
    try {
      msg = (await err.response.json()).error || msg;
    } catch (_) { }
    showNotification(msg, 'danger');
  } finally {
    toggleButtons(true);
  }
}

function buildFormData(file) {
  const fd = new FormData();
  fd.append('file', file);

  const searchPath = $('#search-path').val();
  const createdDate = $('#created-date').val();
  const updatedDate = $('#updated-date').val();
  const fileAuthor = $('#file-author').val();

  if (searchPath) fd.append('search_path', searchPath);
  if (createdDate) fd.append('created_date', createdDate);
  if (updatedDate) fd.append('updated_date', updatedDate);
  if (fileAuthor) fd.append('file_author', fileAuthor);

  categories.forEach(ct => {
    const val = $(`#category-${ct.id}`).val()
    if (val) {
      fd.append(ct.id, val)
    }
  })
  return fd;
}

async function view(id) {
  const data = await fetchJSON(`document_manager/get_content/${id}`);
  buildContentView(data);
  bootstrap.Modal.getOrCreateInstance('#contentModal').show();
}

function buildContentView(data) {
  $('#content').empty();
  if (!data) return;

  data.forEach(a => {
    $('#content').append(`<p class="chunk">${a?.page_content}</p>`);
  });
}
async function delete_file(id) {
  if (!confirm('Are you sure ?')) return;
  const res = await fetch(`${API_BASE}document_manager/${id}`, { method: 'DELETE' });
  if (!res.ok) {
    showNotification("Error while deleting the file", 'error')
    return;
  }
  showNotification("File deleted successully");
  loadPage();

}
// ------------------------ Utilities ------------------------- //

function showNotification(message, type = 'success') {
  const $note = $('#uploadNotification')
    .removeClass('alert-success alert-danger')
    .addClass(`alert-${type}`)
    .text(message)
    .fadeIn(200);

  setTimeout(() => $note.fadeOut(300), 2500);
}

function toggleButtons(enabled) {
  $('button').prop('disabled', !enabled);
}

function clearFields() {
  $('#uploadForm').trigger('reset');
}

function get_category_vals(record) {
  const vals = []
  categories.forEach(cat => {
    const val = record[cat['id']]
    if (val) vals.push(val)
  });
  return vals.join(', ')
}