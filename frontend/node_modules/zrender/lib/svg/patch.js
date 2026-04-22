import { isArray, isObject } from '../core/util.js';
import { createElement, createVNode, XMLNS, XML_NAMESPACE, XLINKNS } from './core.js';
import * as api from './domapi.js';
var colonChar = 58;
var xChar = 120;
var emptyNode = createVNode('', '');
function isUndef(s) {
    return s === undefined;
}
function isDef(s) {
    return s !== undefined;
}
function createKeyToOldIdx(children, beginIdx, endIdx) {
    var map = {};
    for (var i = beginIdx; i <= endIdx; ++i) {
        var key = children[i].key;
        if (key !== undefined) {
            if (process.env.NODE_ENV !== 'production') {
                if (map[key] != null) {
                    console.error("Duplicate key " + key);
                }
            }
            map[key] = i;
        }
    }
    return map;
}
function sameVnode(vnode1, vnode2) {
    var isSameKey = vnode1.key === vnode2.key;
    var isSameTag = vnode1.tag === vnode2.tag;
    return isSameTag && isSameKey;
}
function createElm(vnode) {
    var i;
    var children = vnode.children;
    var tag = vnode.tag;
    if (isDef(tag)) {
        var elm = (vnode.elm = createElement(tag));
        updateAttrs(emptyNode, vnode);
        if (isArray(children)) {
            for (i = 0; i < children.length; ++i) {
                var ch = children[i];
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
        vnode.elm = api.createTextNode(vnode.text);
    }
    return vnode.elm;
}
function addVnodes(parentElm, before, vnodes, startIdx, endIdx) {
    for (; startIdx <= endIdx; ++startIdx) {
        var ch = vnodes[startIdx];
        if (ch != null) {
            api.insertBefore(parentElm, createElm(ch), before);
        }
    }
}
function removeVnodes(parentElm, vnodes, startIdx, endIdx) {
    for (; startIdx <= endIdx; ++startIdx) {
        var ch = vnodes[startIdx];
        if (ch != null) {
            if (isDef(ch.tag)) {
                var parent_1 = api.parentNode(ch.elm);
                api.removeChild(parent_1, ch.elm);
            }
            else {
                api.removeChild(parentElm, ch.elm);
            }
        }
    }
}
export function updateAttrs(oldVnode, vnode) {
    var key;
    var elm = vnode.elm;
    var oldAttrs = oldVnode && oldVnode.attrs || {};
    var attrs = vnode.attrs || {};
    if (oldAttrs === attrs) {
        return;
    }
    for (key in attrs) {
        var cur = attrs[key];
        var old = oldAttrs[key];
        if (old !== cur) {
            if (cur === true) {
                elm.setAttribute(key, '');
            }
            else if (cur === false) {
                elm.removeAttribute(key);
            }
            else {
                if (key === 'style') {
                    elm.style.cssText = cur;
                }
                else if (key.charCodeAt(0) !== xChar) {
                    elm.setAttribute(key, cur);
                }
                else if (key === 'xmlns:xlink' || key === 'xmlns') {
                    elm.setAttributeNS(XMLNS, key, cur);
                }
                else if (key.charCodeAt(3) === colonChar) {
                    elm.setAttributeNS(XML_NAMESPACE, key, cur);
                }
                else if (key.charCodeAt(5) === colonChar) {
                    elm.setAttributeNS(XLINKNS, key, cur);
                }
                else {
                    elm.setAttribute(key, cur);
                }
            }
        }
    }
    for (key in oldAttrs) {
        if (!(key in attrs)) {
            elm.removeAttribute(key);
        }
    }
}
function updateChildren(parentElm, oldCh, newCh) {
    var oldStartIdx = 0;
    var newStartIdx = 0;
    var oldEndIdx = oldCh.length - 1;
    var oldStartVnode = oldCh[0];
    var oldEndVnode = oldCh[oldEndIdx];
    var newEndIdx = newCh.length - 1;
    var newStartVnode = newCh[0];
    var newEndVnode = newCh[newEndIdx];
    var oldKeyToIdx;
    var idxInOld;
    var elmToMove;
    var before;
    while (oldStartIdx <= oldEndIdx && newStartIdx <= newEndIdx) {
        if (oldStartVnode == null) {
            oldStartVnode = oldCh[++oldStartIdx];
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
            patchVnode(oldStartVnode, newEndVnode);
            api.insertBefore(parentElm, oldStartVnode.elm, api.nextSibling(oldEndVnode.elm));
            oldStartVnode = oldCh[++oldStartIdx];
            newEndVnode = newCh[--newEndIdx];
        }
        else if (sameVnode(oldEndVnode, newStartVnode)) {
            patchVnode(oldEndVnode, newStartVnode);
            api.insertBefore(parentElm, oldEndVnode.elm, oldStartVnode.elm);
            oldEndVnode = oldCh[--oldEndIdx];
            newStartVnode = newCh[++newStartIdx];
        }
        else {
            if (isUndef(oldKeyToIdx)) {
                oldKeyToIdx = createKeyToOldIdx(oldCh, oldStartIdx, oldEndIdx);
            }
            idxInOld = oldKeyToIdx[newStartVnode.key];
            if (isUndef(idxInOld)) {
                api.insertBefore(parentElm, createElm(newStartVnode), oldStartVnode.elm);
            }
            else {
                elmToMove = oldCh[idxInOld];
                if (elmToMove.tag !== newStartVnode.tag) {
                    api.insertBefore(parentElm, createElm(newStartVnode), oldStartVnode.elm);
                }
                else {
                    patchVnode(elmToMove, newStartVnode);
                    oldCh[idxInOld] = undefined;
                    api.insertBefore(parentElm, elmToMove.elm, oldStartVnode.elm);
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
function patchVnode(oldVnode, vnode) {
    var elm = (vnode.elm = oldVnode.elm);
    var oldCh = oldVnode.children;
    var ch = vnode.children;
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
        api.setTextContent(elm, vnode.text);
    }
}
export default function patch(oldVnode, vnode) {
    if (sameVnode(oldVnode, vnode)) {
        patchVnode(oldVnode, vnode);
    }
    else {
        var elm = oldVnode.elm;
        var parent_2 = api.parentNode(elm);
        createElm(vnode);
        if (parent_2 !== null) {
            api.insertBefore(parent_2, vnode.elm, api.nextSibling(elm));
            removeVnodes(parent_2, [oldVnode], 0, 0);
        }
    }
    return vnode;
}
