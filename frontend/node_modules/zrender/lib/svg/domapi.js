export function createTextNode(text) {
    return document.createTextNode(text);
}
export function createComment(text) {
    return document.createComment(text);
}
export function insertBefore(parentNode, newNode, referenceNode) {
    parentNode.insertBefore(newNode, referenceNode);
}
export function removeChild(node, child) {
    node.removeChild(child);
}
export function appendChild(node, child) {
    node.appendChild(child);
}
export function parentNode(node) {
    return node.parentNode;
}
export function nextSibling(node) {
    return node.nextSibling;
}
export function tagName(elm) {
    return elm.tagName;
}
export function setTextContent(node, text) {
    node.textContent = text;
}
export function getTextContent(node) {
    return node.textContent;
}
export function isElement(node) {
    return node.nodeType === 1;
}
export function isText(node) {
    return node.nodeType === 3;
}
export function isComment(node) {
    return node.nodeType === 8;
}
