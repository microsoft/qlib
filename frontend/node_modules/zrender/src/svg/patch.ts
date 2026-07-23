/**
 * Virtual DOM patching
 * Modified from snabbdom https://github.com/snabbdom/snabbdom/blob/master/src/init.ts
 *
 * The design has been simplified to focus on the purpose in SVG rendering in SVG.
 *
 * Licensed under the MIT License
 * https://github.com/paldepind/snabbdom/blob/master/LICENSE
 */

import { isArray, isObject } from '../core/util';
import { createElement, createVNode, SVGVNode, XMLNS, XML_NAMESPACE, XLINKNS } from './core';
import * as api from './domapi';

const colonChar = 58;
const xChar = 120;
const emptyNode = createVNode('', '');

type NonUndefined<T> = T extends undefined ? never : T;

function isUndef(s: any): boolean {
    return s === undefined;
}

function isDef<A>(s: A): s is NonUndefined<A> {
    return s !== undefined;
}

function createKeyToOldIdx(
    children: SVGVNode[],
    beginIdx: number,
    endIdx: number
): KeyToIndexMap {
    const map: KeyToIndexMap = {};
    for (let i = beginIdx; i <= endIdx; ++i) {
        const key = children[i].key;
        if (key !== undefined) {
            if (process.env.NODE_ENV !== 'production') {
                if (map[key] != null) {
                    console.error(`Duplicate key ${key}`);
                }
            }
            map[key] = i;
        }
    }
    return map;
}

function sameVnode(vnode1: SVGVNode, vnode2: SVGVNode): boolean {
    const isSameKey = vnode1.key === vnode2.key;
    const isSameTag = vnode1.tag === vnode2.tag;

    return isSameTag && isSameKey;
}

type KeyToIndexMap = { [key: string]: number };

function createElm(vnode: SVGVNode): Node {
    let i: any;
    const children = vnode.children;
    const tag = vnode.tag;
    // if (tag === '!') {
    //     if (isUndef(vnode.text)) {
    //         vnode.text = '';
    //     }
    //     vnode.elm = api.createComment(vnode.text!);
    // }
    // else
    if (isDef(tag)) {
        const elm = (vnode.elm = createElement(tag));

        updateAttrs(emptyNode, vnode);

        if (isArray(children)) {
            for (i = 0; i < children.length; ++i) {
                const ch = children[i];
                if (ch != null) {
                    api.appendChild(elm, createElm(ch));
                }
            }
        }
        else if (isDef(vnode.text) && !isObject(vnode.text)) {
            api.appendChild(elm, api.createTextNode(vnode.text));
        }
    }
    else {
        vnode.elm = api.createTextNode(vnode.text!);
    }
    return vnode.elm;
}

function addVnodes(
    parentElm: Node,
    before: Node | null,
    vnodes: SVGVNode[],
    startIdx: number,
    endIdx: number
) {
    for (; startIdx <= endIdx; ++startIdx) {
        const ch = vnodes[startIdx];
        if (ch != null) {
            api.insertBefore(parentElm, createElm(ch), before);
        }
    }
}

function removeVnodes(parentElm: Node, vnodes: SVGVNode[], startIdx: number, endIdx: number): void {
    for (; startIdx <= endIdx; ++startIdx) {
        const ch = vnodes[startIdx];
        if (ch != null) {
            if (isDef(ch.tag)) {
                const parent = api.parentNode(ch.elm);
                api.removeChild(parent, ch.elm);
            }
            else {
                // Text node
                api.removeChild(parentElm, ch.elm!);
            }
        }
    }
}

export function updateAttrs(oldVnode: SVGVNode, vnode: SVGVNode): void {
    let key: string;
    const elm = vnode.elm as SVGElement;
    const oldAttrs = oldVnode && oldVnode.attrs || {};
    const attrs = vnode.attrs || {};

    if (oldAttrs === attrs) {
        return;
    }

    // update modified attributes, add new attributes
    // eslint-disable-next-line
    for (key in attrs) {
        const cur = attrs[key];
        const old = oldAttrs[key];
        if (old !== cur) {
            if (cur === true) {
                elm.setAttribute(key, '');
            }
            else if (cur === false) {
                elm.removeAttribute(key);
            }
            else {
                if (key === 'style') {
                    elm.style.cssText = cur as string;
                }
                else if (key.charCodeAt(0) !== xChar) {
                    elm.setAttribute(key, cur as any);
                }
                // TODO
                else if (key === 'xmlns:xlink' || key === 'xmlns') {
                    elm.setAttributeNS(XMLNS, key, cur as any);
                }
                else if (key.charCodeAt(3) === colonChar) {
                    // Assume xml namespace
                    elm.setAttributeNS(XML_NAMESPACE, key, cur as any);
                }
                else if (key.charCodeAt(5) === colonChar) {
                    // Assume xlink namespace
                    elm.setAttributeNS(XLINKNS, key, cur as any);
                }
                else {
                    elm.setAttribute(key, cur as any);
                }
            }
        }
    }

    // remove removed attributes
    // use `in` operator since the previous `for` iteration uses it (.i.e. add even attributes with undefined value)
    // the other option is to remove all attributes with value == undefined
    for (key in oldAttrs) {
        if (!(key in attrs)) {
            elm.removeAttribute(key);
        }
    }
}


function updateChildren(parentElm: Node, oldCh: SVGVNode[], newCh: SVGVNode[]) {
    let oldStartIdx = 0;
    let newStartIdx = 0;
    let oldEndIdx = oldCh.length - 1;
    let oldStartVnode = oldCh[0];
    let oldEndVnode = oldCh[oldEndIdx];
    let newEndIdx = newCh.length - 1;
    let newStartVnode = newCh[0];
    let newEndVnode = newCh[newEndIdx];
    let oldKeyToIdx: KeyToIndexMap | undefined;
    let idxInOld: number;
    let elmToMove: SVGVNode;
    let before: any;

    while (oldStartIdx <= oldEndIdx && newStartIdx <= newEndIdx) {
        if (oldStartVnode == null) {
            oldStartVnode = oldCh[++oldStartIdx]; // Vnode might have been moved left
        }
        else if (oldEndVnode == null) {
            oldEndVnode = oldCh[--oldEndIdx];
        }
        else if (newStartVnode == null) {
            newStartVnode = newCh[++newStartIdx];
        }
        else if (newEndVnode == null) {
            newEndVnode = newCh[--newEndIdx];
        }
        else if (sameVnode(oldStartVnode, newStartVnode)) {
            patchVnode(oldStartVnode, newStartVnode);
            oldStartVnode = oldCh[++oldStartIdx];
            newStartVnode = newCh[++newStartIdx];
        }
        else if (sameVnode(oldEndVnode, newEndVnode)) {
            patchVnode(oldEndVnode, newEndVnode);
            oldEndVnode = oldCh[--oldEndIdx];
            newEndVnode = newCh[--newEndIdx];
        }
        else if (sameVnode(oldStartVnode, newEndVnode)) {
            // Vnode moved right
            patchVnode(oldStartVnode, newEndVnode);
            api.insertBefore(parentElm, oldStartVnode.elm!, api.nextSibling(oldEndVnode.elm!));
            oldStartVnode = oldCh[++oldStartIdx];
            newEndVnode = newCh[--newEndIdx];
        }
        else if (sameVnode(oldEndVnode, newStartVnode)) {
            // Vnode moved left
            patchVnode(oldEndVnode, newStartVnode);
            api.insertBefore(parentElm, oldEndVnode.elm!, oldStartVnode.elm!);
            oldEndVnode = oldCh[--oldEndIdx];
            newStartVnode = newCh[++newStartIdx];
        }
        else {
            if (isUndef(oldKeyToIdx)) {
                oldKeyToIdx = createKeyToOldIdx(oldCh, oldStartIdx, oldEndIdx);
            }
            idxInOld = oldKeyToIdx[newStartVnode.key];
            if (isUndef(idxInOld)) {
                // New element
                api.insertBefore(parentElm, createElm(newStartVnode), oldStartVnode.elm!);
            }
            else {
                elmToMove = oldCh[idxInOld];
                if (elmToMove.tag !== newStartVnode.tag) {
                    api.insertBefore(parentElm, createElm(newStartVnode), oldStartVnode.elm!);
                }
                else {
                    patchVnode(elmToMove, newStartVnode);
                    oldCh[idxInOld] = undefined;
                    api.insertBefore(parentElm, elmToMove.elm!, oldStartVnode.elm!);
                }
            }
            newStartVnode = newCh[++newStartIdx];
        }
    }
    if (oldStartIdx <= oldEndIdx || newStartIdx <= newEndIdx) {
        if (oldStartIdx > oldEndIdx) {
            before = newCh[newEndIdx + 1] == null ? null : newCh[newEndIdx + 1].elm;
            addVnodes(parentElm, before, newCh, newStartIdx, newEndIdx);
        }
        else {
            removeVnodes(parentElm, oldCh, oldStartIdx, oldEndIdx);
        }
    }
}

function patchVnode(oldVnode: SVGVNode, vnode: SVGVNode) {
    const elm = (vnode.elm = oldVnode.elm)!;
    const oldCh = oldVnode.children;
    const ch = vnode.children;
    if (oldVnode === vnode) {
        return;
    }

    updateAttrs(oldVnode, vnode);

    if (isUndef(vnode.text)) {
        if (isDef(oldCh) && isDef(ch)) {
            if (oldCh !== ch) {
                updateChildren(elm, oldCh, ch);
            }
        }
        else if (isDef(ch)) {
            if (isDef(oldVnode.text)) {
                api.setTextContent(elm, '');
            }
            addVnodes(elm, null, ch, 0, ch.length - 1);
        }
        else if (isDef(oldCh)) {
            removeVnodes(elm, oldCh, 0, oldCh.length - 1);
        }
        else if (isDef(oldVnode.text)) {
            api.setTextContent(elm, '');
        }
    }
    else if (oldVnode.text !== vnode.text) {
        if (isDef(oldCh)) {
            removeVnodes(elm, oldCh, 0, oldCh.length - 1);
        }
        api.setTextContent(elm, vnode.text!);
    }
}

export default function patch(oldVnode: SVGVNode, vnode: SVGVNode): SVGVNode {
    if (sameVnode(oldVnode, vnode)) {
        patchVnode(oldVnode, vnode);
    }
    else {
        const elm = oldVnode.elm!;
        const parent = api.parentNode(elm);

        createElm(vnode);

        if (parent !== null) {
            api.insertBefore(parent, vnode.elm!, api.nextSibling(elm));
            removeVnodes(parent, [oldVnode], 0, 0);
        }
    }

    return vnode;
}
