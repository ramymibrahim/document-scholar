const LS_ACTIVE = "chat.activeId";
const LS_LIST = "chat.list";
const MAX_CHATS = 50;

function loadList() {
  try { return JSON.parse(localStorage.getItem(LS_LIST) || "[]"); }
  catch { return []; }
}
function saveList(list) {
  localStorage.setItem(LS_LIST, JSON.stringify(list.slice(0, MAX_CHATS)));
}
export function getActiveChatId() {
  return localStorage.getItem(LS_ACTIVE) || null;
}
export function setActiveChatId(id) {
  if (!id) return;
  localStorage.setItem(LS_ACTIVE, id);
}

export function addOrUpdateChat(id) {
  let list = loadList().filter(x => x && x && x !== id);
  list.push(id);
  saveList(list);
  setActiveChatId(id);
  return list;
}

export function removeChat(id) {
  let list = loadList().filter(x => x !== id);
  saveList(list);
  if (getActiveChatId() === id) {
    setActiveChatId(list[0] || "");
  }
  return list;
}

export function getChats() {
  return loadList();
}
