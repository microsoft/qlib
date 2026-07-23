import { keys, map } from '../core/util.js';
import { encodeHTML } from '../core/dom.js';
export var SVGNS = 'http://www.w3.org/2000/svg';
export var XLINKNS = 'http://www.w3.org/1999/xlink';
export var XMLNS = 'http://www.w3.org/2000/xmlns/';
export var XML_NAMESPACE = 'http://www.w3.org/XML/1998/namespace';
export var META_DATA_PREFIX = 'ecmeta_';
export function createElement(name) {
    return document.createElementNS(SVGNS, name);
}
;
export function createVNode(tag, key, attrs, children, text) {
    return {
        tag: tag,
        attrs: attrs || {},
        children: children,
        text: text,
        key: key
    };
}
function createElementOpen(name, attrs) {
    var attrsStr = [];
    if (attrs) {
        for (var key in attrs) {
            var val = attrs[key];
            var part = key;
            if (val === false) {
                continue;
            }
            else if (val !== true && val != null) {
                part += "=\"" + val + "\"";
            }
            attrsStr.push(part);
        }
    }
    return "<" + name + " " + attrsStr.join(' ') + ">";
}
function createElementClose(name) {
    return "</" + name + ">";
}
export function vNodeToString(el, opts) {
    opts = opts || {};
    var S = opts.newline ? '\n' : '';
    function convertElToString(el) {
        var children = el.children, tag = el.tag, attrs = el.attrs, text = el.text;
        return createElementOpen(tag, attrs)
            + (tag !== 'style' ? encodeHTML(text) : text || '')
            + (children ? "" + S + map(children, function (child) { return convertElToString(child); }).join(S) + S : '')
            + createElementClose(tag);
    }
    return convertElToString(el);
}
export function getCssString(selectorNodes, animationNodes, opts) {
    opts = opts || {};
    var S = opts.newline ? '\n' : '';
    var bracketBegin = " {" + S;
    var bracketEnd = S + "}";
    var selectors = map(keys(selectorNodes), function (className) {
        return className + bracketBegin + map(keys(selectorNodes[className]), function (attrName) {
            return attrName + ":" + selectorNodes[className][attrName] + ";";
        }).join(S) + bracketEnd;
    }).join(S);
    var animations = map(keys(animationNodes), function (animationName) {
        return "@keyframes " + animationName + bracketBegin + map(keys(animationNodes[animationName]), function (percent) {
            return percent + bracketBegin + map(keys(animationNodes[animationName][percent]), function (attrName) {
                var val = animationNodes[animationName][percent][attrName];
                if (attrName === 'd') {
                    val = "path(\"" + val + "\")";
                }
                return attrName + ":" + val + ";";
            }).join(S) + bracketEnd;
        }).join(S) + bracketEnd;
    }).join(S);
    if (!selectors && !animations) {
        return '';
    }
    return ['<![CDATA[', selectors, animations, ']]>'].join(S);
}
export function createBrushScope(zrId) {
    return {
        zrId: zrId,
        shadowCache: {},
        patternCache: {},
        gradientCache: {},
        clipPathCache: {},
        defs: {},
        cssNodes: {},
        cssAnims: {},
        cssStyleCache: {},
        cssAnimIdx: 0,
        shadowIdx: 0,
        gradientIdx: 0,
        patternIdx: 0,
        clipPathIdx: 0
    };
}
export function createSVGVNode(width, height, children, useViewBox) {
    return createVNode('svg', 'root', {
        'width': width,
        'height': height,
        'xmlns': SVGNS,
        'xmlns:xlink': XLINKNS,
        'version': '1.1',
        'baseProfile': 'full',
        'viewBox': useViewBox ? "0 0 " + width + " " + height : false
    }, children);
}
