import Group from '../graphic/Group.js';
import ZRImage from '../graphic/Image.js';
import Circle from '../graphic/shape/Circle.js';
import Rect from '../graphic/shape/Rect.js';
import Ellipse from '../graphic/shape/Ellipse.js';
import Line from '../graphic/shape/Line.js';
import Polygon from '../graphic/shape/Polygon.js';
import Polyline from '../graphic/shape/Polyline.js';
import * as matrix from '../core/matrix.js';
import { createFromString } from './path.js';
import { defaults, trim, each, map, keys, hasOwn } from '../core/util.js';
import LinearGradient from '../graphic/LinearGradient.js';
import RadialGradient from '../graphic/RadialGradient.js';
import TSpan from '../graphic/TSpan.js';
import { parseXML } from './parseXML.js';
;
var nodeParsers;
var INHERITABLE_STYLE_ATTRIBUTES_MAP = {
    'fill': 'fill',
    'stroke': 'stroke',
    'stroke-width': 'lineWidth',
    'opacity': 'opacity',
    'fill-opacity': 'fillOpacity',
    'stroke-opacity': 'strokeOpacity',
    'stroke-dasharray': 'lineDash',
    'stroke-dashoffset': 'lineDashOffset',
    'stroke-linecap': 'lineCap',
    'stroke-linejoin': 'lineJoin',
    'stroke-miterlimit': 'miterLimit',
    'font-family': 'fontFamily',
    'font-size': 'fontSize',
    'font-style': 'fontStyle',
    'font-weight': 'fontWeight',
    'text-anchor': 'textAlign',
    'visibility': 'visibility',
    'display': 'display'
};
var INHERITABLE_STYLE_ATTRIBUTES_MAP_KEYS = keys(INHERITABLE_STYLE_ATTRIBUTES_MAP);
var SELF_STYLE_ATTRIBUTES_MAP = {
    'alignment-baseline': 'textBaseline',
    'stop-color': 'stopColor'
};
var SELF_STYLE_ATTRIBUTES_MAP_KEYS = keys(SELF_STYLE_ATTRIBUTES_MAP);
var SVGParser = (function () {
    function SVGParser() {
        this._defs = {};
        this._root = null;
    }
    SVGParser.prototype.parse = function (xml, opt) {
        opt = opt || {};
        var svg = parseXML(xml);
        if (process.env.NODE_ENV !== 'production') {
            if (!svg) {
                throw new Error('Illegal svg');
            }
        }
        this._defsUsePending = [];
        var root = new Group();
        this._root = root;
        var named = [];
        var viewBox = svg.getAttribute('viewBox') || '';
        var width = parseFloat((svg.getAttribute('width') || opt.width));
        var height = parseFloat((svg.getAttribute('height') || opt.height));
        isNaN(width) && (width = null);
        isNaN(height) && (height = null);
        parseAttributes(svg, root, null, true, false);
        var child = svg.firstChild;
        while (child) {
            this._parseNode(child, root, named, null, false, false);
            child = child.nextSibling;
        }
        applyDefs(this._defs, this._defsUsePending);
        this._defsUsePending = [];
        var viewBoxRect;
        var viewBoxTransform;
        if (viewBox) {
            var viewBoxArr = splitNumberSequence(viewBox);
            if (viewBoxArr.length >= 4) {
                viewBoxRect = {
                    x: parseFloat((viewBoxArr[0] || 0)),
                    y: parseFloat((viewBoxArr[1] || 0)),
                    width: parseFloat(viewBoxArr[2]),
                    height: parseFloat(viewBoxArr[3])
                };
            }
        }
        if (viewBoxRect && width != null && height != null) {
            viewBoxTransform = makeViewBoxTransform(viewBoxRect, { x: 0, y: 0, width: width, height: height });
            if (!opt.ignoreViewBox) {
                var elRoot = root;
                root = new Group();
                root.add(elRoot);
                elRoot.scaleX = elRoot.scaleY = viewBoxTransform.scale;
                elRoot.x = viewBoxTransform.x;
                elRoot.y = viewBoxTransform.y;
            }
        }
        if (!opt.ignoreRootClip && width != null && height != null) {
            root.setClipPath(new Rect({
                shape: { x: 0, y: 0, width: width, height: height }
            }));
        }
        return {
            root: root,
            width: width,
            height: height,
            viewBoxRect: viewBoxRect,
            viewBoxTransform: viewBoxTransform,
            named: named
        };
    };
    SVGParser.prototype._parseNode = function (xmlNode, parentGroup, named, namedFrom, isInDefs, isInText) {
        var nodeName = xmlNode.nodeName.toLowerCase();
        var el;
        var namedFromForSub = namedFrom;
        if (nodeName === 'defs') {
            isInDefs = true;
        }
        if (nodeName === 'text') {
            isInText = true;
        }
        if (nodeName === 'defs' || nodeName === 'switch') {
            el = parentGroup;
        }
        else {
            if (!isInDefs) {
                var parser_1 = nodeParsers[nodeName];
                if (parser_1 && hasOwn(nodeParsers, nodeName)) {
                    el = parser_1.call(this, xmlNode, parentGroup);
                    var nameAttr = xmlNode.getAttribute('name');
                    if (nameAttr) {
                        var newNamed = {
                            name: nameAttr,
                            namedFrom: null,
                            svgNodeTagLower: nodeName,
                            el: el
                        };
                        named.push(newNamed);
                        if (nodeName === 'g') {
                            namedFromForSub = newNamed;
                        }
                    }
                    else if (namedFrom) {
                        named.push({
                            name: namedFrom.name,
                            namedFrom: namedFrom,
                            svgNodeTagLower: nodeName,
                            el: el
                        });
                    }
                    parentGroup.add(el);
                }
            }
            var parser = paintServerParsers[nodeName];
            if (parser && hasOwn(paintServerParsers, nodeName)) {
                var def = parser.call(this, xmlNode);
                var id = xmlNode.getAttribute('id');
                if (id) {
                    this._defs[id] = def;
                }
            }
        }
        if (el && el.isGroup) {
            var child = xmlNode.firstChild;
            while (child) {
                if (child.nodeType === 1) {
                    this._parseNode(child, el, named, namedFromForSub, isInDefs, isInText);
                }
                else if (child.nodeType === 3 && isInText) {
                    this._parseText(child, el);
                }
                child = child.nextSibling;
            }
        }
    };
    SVGParser.prototype._parseText = function (xmlNode, parentGroup) {
        var text = new TSpan({
            style: {
                text: xmlNode.textContent
            },
            silent: true,
            x: this._textX || 0,
            y: this._textY || 0
        });
        inheritStyle(parentGroup, text);
        parseAttributes(xmlNode, text, this._defsUsePending, false, false);
        applyTextAlignment(text, parentGroup);
        var textStyle = text.style;
        var fontSize = textStyle.fontSize;
        if (fontSize && fontSize < 9) {
            textStyle.fontSize = 9;
            text.scaleX *= fontSize / 9;
            text.scaleY *= fontSize / 9;
        }
        var font = (textStyle.fontSize || textStyle.fontFamily) && [
            textStyle.fontStyle,
            textStyle.fontWeight,
            (textStyle.fontSize || 12) + 'px',
            textStyle.fontFamily || 'sans-serif'
        ].join(' ');
        textStyle.font = font;
        var rect = text.getBoundingRect();
        this._textX += rect.width;
        parentGroup.add(text);
        return text;
    };
    SVGParser.internalField = (function () {
        nodeParsers = {
            'g': function (xmlNode, parentGroup) {
                var g = new Group();
                inheritStyle(parentGroup, g);
                parseAttributes(xmlNode, g, this._defsUsePending, false, false);
                return g;
            },
            'rect': function (xmlNode, parentGroup) {
                var rect = new Rect();
                inheritStyle(parentGroup, rect);
                parseAttributes(xmlNode, rect, this._defsUsePending, false, false);
                rect.setShape({
                    x: parseFloat(xmlNode.getAttribute('x') || '0'),
                    y: parseFloat(xmlNode.getAttribute('y') || '0'),
                    width: parseFloat(xmlNode.getAttribute('width') || '0'),
                    height: parseFloat(xmlNode.getAttribute('height') || '0')
                });
                rect.silent = true;
                return rect;
            },
            'circle': function (xmlNode, parentGroup) {
                var circle = new Circle();
                inheritStyle(parentGroup, circle);
                parseAttributes(xmlNode, circle, this._defsUsePending, false, false);
                circle.setShape({
                    cx: parseFloat(xmlNode.getAttribute('cx') || '0'),
                    cy: parseFloat(xmlNode.getAttribute('cy') || '0'),
                    r: parseFloat(xmlNode.getAttribute('r') || '0')
                });
                circle.silent = true;
                return circle;
            },
            'line': function (xmlNode, parentGroup) {
                var line = new Line();
                inheritStyle(parentGroup, line);
                parseAttributes(xmlNode, line, this._defsUsePending, false, false);
                line.setShape({
                    x1: parseFloat(xmlNode.getAttribute('x1') || '0'),
                    y1: parseFloat(xmlNode.getAttribute('y1') || '0'),
                    x2: parseFloat(xmlNode.getAttribute('x2') || '0'),
                    y2: parseFloat(xmlNode.getAttribute('y2') || '0')
                });
                line.silent = true;
                return line;
            },
            'ellipse': function (xmlNode, parentGroup) {
                var ellipse = new Ellipse();
                inheritStyle(parentGroup, ellipse);
                parseAttributes(xmlNode, ellipse, this._defsUsePending, false, false);
                ellipse.setShape({
                    cx: parseFloat(xmlNode.getAttribute('cx') || '0'),
                    cy: parseFloat(xmlNode.getAttribute('cy') || '0'),
                    rx: parseFloat(xmlNode.getAttribute('rx') || '0'),
                    ry: parseFloat(xmlNode.getAttribute('ry') || '0')
                });
                ellipse.silent = true;
                return ellipse;
            },
            'polygon': function (xmlNode, parentGroup) {
                var pointsStr = xmlNode.getAttribute('points');
                var pointsArr;
                if (pointsStr) {
                    pointsArr = parsePoints(pointsStr);
                }
                var polygon = new Polygon({
                    shape: {
                        points: pointsArr || []
                    },
                    silent: true
                });
                inheritStyle(parentGroup, polygon);
                parseAttributes(xmlNode, polygon, this._defsUsePending, false, false);
                return polygon;
            },
            'polyline': function (xmlNode, parentGroup) {
                var pointsStr = xmlNode.getAttribute('points');
                var pointsArr;
                if (pointsStr) {
                    pointsArr = parsePoints(pointsStr);
                }
                var polyline = new Polyline({
                    shape: {
                        points: pointsArr || []
                    },
                    silent: true
                });
                inheritStyle(parentGroup, polyline);
                parseAttributes(xmlNode, polyline, this._defsUsePending, false, false);
                return polyline;
            },
            'image': function (xmlNode, parentGroup) {
                var img = new ZRImage();
                inheritStyle(parentGroup, img);
                parseAttributes(xmlNode, img, this._defsUsePending, false, false);
                img.setStyle({
                    image: xmlNode.getAttribute('xlink:href') || xmlNode.getAttribute('href'),
                    x: +xmlNode.getAttribute('x'),
                    y: +xmlNode.getAttribute('y'),
                    width: +xmlNode.getAttribute('width'),
                    height: +xmlNode.getAttribute('height')
                });
                img.silent = true;
                return img;
            },
            'text': function (xmlNode, parentGroup) {
                var x = xmlNode.getAttribute('x') || '0';
                var y = xmlNode.getAttribute('y') || '0';
                var dx = xmlNode.getAttribute('dx') || '0';
                var dy = xmlNode.getAttribute('dy') || '0';
                this._textX = parseFloat(x) + parseFloat(dx);
                this._textY = parseFloat(y) + parseFloat(dy);
                var g = new Group();
                inheritStyle(parentGroup, g);
                parseAttributes(xmlNode, g, this._defsUsePending, false, true);
                return g;
            },
            'tspan': function (xmlNode, parentGroup) {
                var x = xmlNode.getAttribute('x');
                var y = xmlNode.getAttribute('y');
                if (x != null) {
                    this._textX = parseFloat(x);
                }
                if (y != null) {
                    this._textY = parseFloat(y);
                }
                var dx = xmlNode.getAttribute('dx') || '0';
                var dy = xmlNode.getAttribute('dy') || '0';
                var g = new Group();
                inheritStyle(parentGroup, g);
                parseAttributes(xmlNode, g, this._defsUsePending, false, true);
                this._textX += parseFloat(dx);
                this._textY += parseFloat(dy);
                return g;
            },
            'path': function (xmlNode, parentGroup) {
                var d = xmlNode.getAttribute('d') || '';
                var path = createFromString(d);
                inheritStyle(parentGroup, path);
                parseAttributes(xmlNode, path, this._defsUsePending, false, false);
                path.silent = true;
                return path;
            }
        };
    })();
    return SVGParser;
}());
var paintServerParsers = {
    'lineargradient': function (xmlNode) {
        var x1 = parseInt(xmlNode.getAttribute('x1') || '0', 10);
        var y1 = parseInt(xmlNode.getAttribute('y1') || '0', 10);
        var x2 = parseInt(xmlNode.getAttribute('x2') || '10', 10);
        var y2 = parseInt(xmlNode.getAttribute('y2') || '0', 10);
        var gradient = new LinearGradient(x1, y1, x2, y2);
        parsePaintServerUnit(xmlNode, gradient);
        parseGradientColorStops(xmlNode, gradient);
        return gradient;
    },
    'radialgradient': function (xmlNode) {
        var cx = parseInt(xmlNode.getAttribute('cx') || '0', 10);
        var cy = parseInt(xmlNode.getAttribute('cy') || '0', 10);
        var r = parseInt(xmlNode.getAttribute('r') || '0', 10);
        var gradient = new RadialGradient(cx, cy, r);
        parsePaintServerUnit(xmlNode, gradient);
        parseGradientColorStops(xmlNode, gradient);
        return gradient;
    }
};
function parsePaintServerUnit(xmlNode, gradient) {
    var gradientUnits = xmlNode.getAttribute('gradientUnits');
    if (gradientUnits === 'userSpaceOnUse') {
        gradient.global = true;
    }
}
function parseGradientColorStops(xmlNode, gradient) {
    var stop = xmlNode.firstChild;
    while (stop) {
        if (stop.nodeType === 1
            && stop.nodeName.toLocaleLowerCase() === 'stop') {
            var offsetStr = stop.getAttribute('offset');
            var offset = void 0;
            if (offsetStr && offsetStr.indexOf('%') > 0) {
                offset = parseInt(offsetStr, 10) / 100;
            }
            else if (offsetStr) {
                offset = parseFloat(offsetStr);
            }
            else {
                offset = 0;
            }
            var styleVals = {};
            parseInlineStyle(stop, styleVals, styleVals);
            var stopColor = styleVals.stopColor
                || stop.getAttribute('stop-color')
                || '#000000';
            gradient.colorStops.push({
                offset: offset,
                color: stopColor
            });
        }
        stop = stop.nextSibling;
    }
}
function inheritStyle(parent, child) {
    if (parent && parent.__inheritedStyle) {
        if (!child.__inheritedStyle) {
            child.__inheritedStyle = {};
        }
        defaults(child.__inheritedStyle, parent.__inheritedStyle);
    }
}
function parsePoints(pointsString) {
    var list = splitNumberSequence(pointsString);
    var points = [];
    for (var i = 0; i < list.length; i += 2) {
        var x = parseFloat(list[i]);
        var y = parseFloat(list[i + 1]);
        points.push([x, y]);
    }
    return points;
}
function parseAttributes(xmlNode, el, defsUsePending, onlyInlineStyle, isTextGroup) {
    var disp = el;
    var inheritedStyle = disp.__inheritedStyle = disp.__inheritedStyle || {};
    var selfStyle = {};
    if (xmlNode.nodeType === 1) {
        parseTransformAttribute(xmlNode, el);
        parseInlineStyle(xmlNode, inheritedStyle, selfStyle);
        if (!onlyInlineStyle) {
            parseAttributeStyle(xmlNode, inheritedStyle, selfStyle);
        }
    }
    disp.style = disp.style || {};
    if (inheritedStyle.fill != null) {
        disp.style.fill = getFillStrokeStyle(disp, 'fill', inheritedStyle.fill, defsUsePending);
    }
    if (inheritedStyle.stroke != null) {
        disp.style.stroke = getFillStrokeStyle(disp, 'stroke', inheritedStyle.stroke, defsUsePending);
    }
    each([
        'lineWidth', 'opacity', 'fillOpacity', 'strokeOpacity', 'miterLimit', 'fontSize'
    ], function (propName) {
        if (inheritedStyle[propName] != null) {
            disp.style[propName] = parseFloat(inheritedStyle[propName]);
        }
    });
    each([
        'lineDashOffset', 'lineCap', 'lineJoin', 'fontWeight', 'fontFamily', 'fontStyle', 'textAlign'
    ], function (propName) {
        if (inheritedStyle[propName] != null) {
            disp.style[propName] = inheritedStyle[propName];
        }
    });
    if (isTextGroup) {
        disp.__selfStyle = selfStyle;
    }
    if (inheritedStyle.lineDash) {
        disp.style.lineDash = map(splitNumberSequence(inheritedStyle.lineDash), function (str) {
            return parseFloat(str);
        });
    }
    if (inheritedStyle.visibility === 'hidden' || inheritedStyle.visibility === 'collapse') {
        disp.invisible = true;
    }
    if (inheritedStyle.display === 'none') {
        disp.ignore = true;
    }
}
function applyTextAlignment(text, parentGroup) {
    var parentSelfStyle = parentGroup.__selfStyle;
    if (parentSelfStyle) {
        var textBaseline = parentSelfStyle.textBaseline;
        var zrTextBaseline = textBaseline;
        if (!textBaseline || textBaseline === 'auto') {
            zrTextBaseline = 'alphabetic';
        }
        else if (textBaseline === 'baseline') {
            zrTextBaseline = 'alphabetic';
        }
        else if (textBaseline === 'before-edge' || textBaseline === 'text-before-edge') {
            zrTextBaseline = 'top';
        }
        else if (textBaseline === 'after-edge' || textBaseline === 'text-after-edge') {
            zrTextBaseline = 'bottom';
        }
        else if (textBaseline === 'central' || textBaseline === 'mathematical') {
            zrTextBaseline = 'middle';
        }
        text.style.textBaseline = zrTextBaseline;
    }
    var parentInheritedStyle = parentGroup.__inheritedStyle;
    if (parentInheritedStyle) {
        var textAlign = parentInheritedStyle.textAlign;
        var zrTextAlign = textAlign;
        if (textAlign) {
            if (textAlign === 'middle') {
                zrTextAlign = 'center';
            }
            text.style.textAlign = zrTextAlign;
        }
    }
}
var urlRegex = /^url\(\s*#(.*?)\)/;
function getFillStrokeStyle(el, method, str, defsUsePending) {
    var urlMatch = str && str.match(urlRegex);
    if (urlMatch) {
        var url = trim(urlMatch[1]);
        defsUsePending.push([el, method, url]);
        return;
    }
    if (str === 'none') {
        str = null;
    }
    return str;
}
function applyDefs(defs, defsUsePending) {
    for (var i = 0; i < defsUsePending.length; i++) {
        var item = defsUsePending[i];
        item[0].style[item[1]] = defs[item[2]];
    }
}
var numberReg = /-?([0-9]*\.)?[0-9]+([eE]-?[0-9]+)?/g;
function splitNumberSequence(rawStr) {
    return rawStr.match(numberReg) || [];
}
var transformRegex = /(translate|scale|rotate|skewX|skewY|matrix)\(([\-\s0-9\.eE,]*)\)/g;
var DEGREE_TO_ANGLE = Math.PI / 180;
function parseTransformAttribute(xmlNode, node) {
    var transform = xmlNode.getAttribute('transform');
    if (transform) {
        transform = transform.replace(/,/g, ' ');
        var transformOps_1 = [];
        var mt = null;
        transform.replace(transformRegex, function (str, type, value) {
            transformOps_1.push(type, value);
            return '';
        });
        for (var i = transformOps_1.length - 1; i > 0; i -= 2) {
            var value = transformOps_1[i];
            var type = transformOps_1[i - 1];
            var valueArr = splitNumberSequence(value);
            mt = mt || matrix.create();
            switch (type) {
                case 'translate':
                    matrix.translate(mt, mt, [parseFloat(valueArr[0]), parseFloat(valueArr[1] || '0')]);
                    break;
                case 'scale':
                    matrix.scale(mt, mt, [parseFloat(valueArr[0]), parseFloat(valueArr[1] || valueArr[0])]);
                    break;
                case 'rotate':
                    matrix.rotate(mt, mt, -parseFloat(valueArr[0]) * DEGREE_TO_ANGLE, [
                        parseFloat(valueArr[1] || '0'),
                        parseFloat(valueArr[2] || '0')
                    ]);
                    break;
                case 'skewX':
                    var sx = Math.tan(parseFloat(valueArr[0]) * DEGREE_TO_ANGLE);
                    matrix.mul(mt, [1, 0, sx, 1, 0, 0], mt);
                    break;
                case 'skewY':
                    var sy = Math.tan(parseFloat(valueArr[0]) * DEGREE_TO_ANGLE);
                    matrix.mul(mt, [1, sy, 0, 1, 0, 0], mt);
                    break;
                case 'matrix':
                    mt[0] = parseFloat(valueArr[0]);
                    mt[1] = parseFloat(valueArr[1]);
                    mt[2] = parseFloat(valueArr[2]);
                    mt[3] = parseFloat(valueArr[3]);
                    mt[4] = parseFloat(valueArr[4]);
                    mt[5] = parseFloat(valueArr[5]);
                    break;
            }
        }
        node.setLocalTransform(mt);
    }
}
var styleRegex = /([^\s:;]+)\s*:\s*([^:;]+)/g;
function parseInlineStyle(xmlNode, inheritableStyleResult, selfStyleResult) {
    var style = xmlNode.getAttribute('style');
    if (!style) {
        return;
    }
    styleRegex.lastIndex = 0;
    var styleRegResult;
    while ((styleRegResult = styleRegex.exec(style)) != null) {
        var svgStlAttr = styleRegResult[1];
        var zrInheritableStlAttr = hasOwn(INHERITABLE_STYLE_ATTRIBUTES_MAP, svgStlAttr)
            ? INHERITABLE_STYLE_ATTRIBUTES_MAP[svgStlAttr]
            : null;
        if (zrInheritableStlAttr) {
            inheritableStyleResult[zrInheritableStlAttr] = styleRegResult[2];
        }
        var zrSelfStlAttr = hasOwn(SELF_STYLE_ATTRIBUTES_MAP, svgStlAttr)
            ? SELF_STYLE_ATTRIBUTES_MAP[svgStlAttr]
            : null;
        if (zrSelfStlAttr) {
            selfStyleResult[zrSelfStlAttr] = styleRegResult[2];
        }
    }
}
function parseAttributeStyle(xmlNode, inheritableStyleResult, selfStyleResult) {
    for (var i = 0; i < INHERITABLE_STYLE_ATTRIBUTES_MAP_KEYS.length; i++) {
        var svgAttrName = INHERITABLE_STYLE_ATTRIBUTES_MAP_KEYS[i];
        var attrValue = xmlNode.getAttribute(svgAttrName);
        if (attrValue != null) {
            inheritableStyleResult[INHERITABLE_STYLE_ATTRIBUTES_MAP[svgAttrName]] = attrValue;
        }
    }
    for (var i = 0; i < SELF_STYLE_ATTRIBUTES_MAP_KEYS.length; i++) {
        var svgAttrName = SELF_STYLE_ATTRIBUTES_MAP_KEYS[i];
        var attrValue = xmlNode.getAttribute(svgAttrName);
        if (attrValue != null) {
            selfStyleResult[SELF_STYLE_ATTRIBUTES_MAP[svgAttrName]] = attrValue;
        }
    }
}
export function makeViewBoxTransform(viewBoxRect, boundingRect) {
    var scaleX = boundingRect.width / viewBoxRect.width;
    var scaleY = boundingRect.height / viewBoxRect.height;
    var scale = Math.min(scaleX, scaleY);
    return {
        scale: scale,
        x: -(viewBoxRect.x + viewBoxRect.width / 2) * scale + (boundingRect.x + boundingRect.width / 2),
        y: -(viewBoxRect.y + viewBoxRect.height / 2) * scale + (boundingRect.y + boundingRect.height / 2)
    };
}
export function parseSVG(xml, opt) {
    var parser = new SVGParser();
    return parser.parse(xml, opt);
}
export { parseXML };
