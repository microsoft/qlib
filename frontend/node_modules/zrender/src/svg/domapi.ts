export function createTextNode(text: string): Text {
    return document.createTextNode(text);
}

export function createComment(text: string): Comment {
    return document.createComment(text);
}

export function insertBefore(
    parentNode: Node,
    newNode: Node,
    referenceNode: Node | null
): void {
    parentNode.insertBefore(newNode, referenceNode);
}

export function removeChild(node: Node, child: Node): void {
    node.removeChild(child);
}

export function appendChild(node: Node, child: Node): void {
    node.appendChild(child);
}

export function parentNode(node: Node): Node | null {
    return node.parentNode;
}

export function nextSibling(node: Node): Node | null {
    return node.nextSibling;
}

export function tagName(elm: Element): string {
    return elm.tagName;
}

export function setTextContent(node: Node, text: string | null): void {
    node.textContent = text;
}

export function getTextContent(node: Node): string | null {
    return node.textContent;
}

export function isElement(node: Node): node is Element {
    return node.nodeType === 1;
}

export function isText(node: Node): node is Text {
    return node.nodeType === 3;
}

export function isComment(node: Node): node is Comment {
    return node.nodeType === 8;
}
