import Group from '../graphic/Group';
import ZRImage from '../graphic/Image';
import Circle from '../graphic/shape/Circle';
import Rect from '../graphic/shape/Rect';
import Ellipse from '../graphic/shape/Ellipse';
import Line from '../graphic/shape/Line';
import Polygon from '../graphic/shape/Polygon';
import Polyline from '../graphic/shape/Polyline';
import * as matrix from '../core/matrix';
import { createFromString } from './path';
import { defaults, trim, each, map, keys, hasOwn } from '../core/util';
import Displayable from '../graphic/Displayable';
import Element from '../Element';
import { RectLike } from '../core/BoundingRect';
import { Dictionary } from '../core/types';
import { PatternObject } from '../graphic/Pattern';
import LinearGradient, { LinearGradientObject } from '../graphic/LinearGradient';
import RadialGradient, { RadialGradientObject } from '../graphic/RadialGradient';
import Gradient, { GradientObject } from '../graphic/Gradient';
import TSpan, { TSpanStyleProps } from '../graphic/TSpan';
import { parseXML } from './parseXML';


interface SVGParserOption {
    // Default width if svg width not specified or is a percent value.
    width?: number;
    // Default height if svg height not specified or is a percent value.
    height?: number;
    ignoreViewBox?: boolean;
    ignoreRootClip?: boolean;
}

export interface SVGParserResult {
    // Group, The root of the the result tree of zrender shapes
    root: Group;
    // number, the viewport width of the SVG
    width: number;
    // number, the viewport height of the SVG
    height: number;
    //  {x, y, width, height}, the declared viewBox rect of the SVG, if exists
    viewBoxRect: RectLike;
    // the {scale, position} calculated by viewBox and viewport, is exists
    viewBoxTransform: {
        x: number;
        y: number;
        scale: number;
    };
    named: SVGParserResultNamedItem[];
}
export interface SVGParserResultNamedItem {
    name: string;
    // If a tag has no name attribute but its ancester <g> is named,
    // `namedFrom` is set to the named item of the ancester <g>.
    // Otherwise null/undefined
    namedFrom: SVGParserResultNamedItem;
    svgNodeTagLower: SVGNodeTagLower;
    el: Element;
};

export type SVGNodeTagLower =
    'g' | 'rect' | 'circle' | 'line' | 'ellipse' | 'polygon'
    | 'polyline' | 'image' | 'text' | 'tspan' | 'path' | 'defs' | 'switch';


type DefsId = string;
type DefsMap = { [id in DefsId]: LinearGradientObject | RadialGradientObject | PatternObject };
type DefsUsePending = [Displayable, 'fill' | 'stroke', DefsId][];

type ElementExtended = Element & {
    __inheritedStyle?: InheritedStyleByZRKey;
    __selfStyle?: SelfStyleByZRKey;
}
type DisplayableExtended = Displayable & {
    __inheritedStyle?: InheritedStyleByZRKey;
    __selfStyle?: SelfStyleByZRKey;
}

type TextStyleOptionExtended = TSpanStyleProps & {
    fontSize: number;
    fontFamily: string;
    fontWeight: string;
    fontStyle: string;
}
let nodeParsers: {[name in SVGNodeTagLower]?: (
    this: SVGParser, xmlNode: SVGElement, parentGroup: Group
) => Element};

type InheritedStyleByZRKey = {[name in InheritableStyleZRKey]?: string};
type InheritableStyleZRKey =
    typeof INHERITABLE_STYLE_ATTRIBUTES_MAP[keyof typeof INHERITABLE_STYLE_ATTRIBUTES_MAP];
const INHERITABLE_STYLE_ATTRIBUTES_MAP = {
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
} as const;
const INHERITABLE_STYLE_ATTRIBUTES_MAP_KEYS = keys(INHERITABLE_STYLE_ATTRIBUTES_MAP);

type SelfStyleByZRKey = {[name in SelfStyleZRKey]?: string};
type SelfStyleZRKey =
    typeof SELF_STYLE_ATTRIBUTES_MAP[keyof typeof SELF_STYLE_ATTRIBUTES_MAP];
const SELF_STYLE_ATTRIBUTES_MAP = {
    'alignment-baseline': 'textBaseline',
    'stop-color': 'stopColor'
};
const SELF_STYLE_ATTRIBUTES_MAP_KEYS = keys(SELF_STYLE_ATTRIBUTES_MAP);


class SVGParser {

    private _defs: DefsMap = {};
    // The use of <defs> can be in front of <defs> declared.
    // So save them temporarily in `_defsUsePending`.
    private _defsUsePending: DefsUsePending;
    private _root: Group = null;

    private _textX: number;
    private _textY: number;

    parse(xml: string | Document | SVGElement, opt: SVGParserOption): SVGParserResult {
        opt = opt || {};

        const svg = parseXML(xml);

        if (process.env.NODE_ENV !== 'production') {
            if (!svg) {
                throw new Error('Illegal svg');
            }
        }

        this._defsUsePending = [];
        let root = new Group();
        this._root = root;
        const named: SVGParserResult['named'] = [];
        // parse view port
        const viewBox = svg.getAttribute('viewBox') || '';

        // If width/height not specified, means "100%" of `opt.width/height`.
        // TODO: Other percent value not supported yet.
        let width = parseFloat((svg.getAttribute('width') || opt.width) as string);
        let height = parseFloat((svg.getAttribute('height') || opt.height) as string);
        // If width/height not specified, set as null for output.
        isNaN(width) && (width = null);
        isNaN(height) && (height = null);

        // Apply inline style on svg element.
        parseAttributes(svg, root, null, true, false);

        let child = svg.firstChild as SVGElement;
        while (child) {
            this._parseNode(child, root, named, null, false, false);
            child = child.nextSibling as SVGElement;
        }

        applyDefs(this._defs, this._defsUsePending);
        this._defsUsePending = [];

        let viewBoxRect;
        let viewBoxTransform;

        if (viewBox) {
            const viewBoxArr = splitNumberSequence(viewBox);
            // Some invalid case like viewBox: 'none'.
            if (viewBoxArr.length >= 4) {
                viewBoxRect = {
                    x: parseFloat((viewBoxArr[0] || 0) as string),
                    y: parseFloat((viewBoxArr[1] || 0) as string),
                    width: parseFloat(viewBoxArr[2]),
                    height: parseFloat(viewBoxArr[3])
                };
            }
        }

        if (viewBoxRect && width != null && height != null) {
            viewBoxTransform = makeViewBoxTransform(viewBoxRect, { x: 0, y: 0, width: width, height: height });

            if (!opt.ignoreViewBox) {
                // If set transform on the output group, it probably bring trouble when
                // some users only intend to show the clipped content inside the viewBox,
                // but not intend to transform the output group. So we keep the output
                // group no transform. If the user intend to use the viewBox as a
                // camera, just set `opt.ignoreViewBox` as `true` and set transfrom
                // manually according to the viewBox info in the output of this method.
                const elRoot = root;
                root = new Group();
                root.add(elRoot);
                elRoot.scaleX = elRoot.scaleY = viewBoxTransform.scale;
                elRoot.x = viewBoxTransform.x;
                elRoot.y = viewBoxTransform.y;
            }
        }

        // Some shapes might be overflow the viewport, which should be
        // clipped despite whether the viewBox is used, as the SVG does.
        if (!opt.ignoreRootClip && width != null && height != null) {
            root.setClipPath(new Rect({
                shape: {x: 0, y: 0, width: width, height: height}
            }));
        }

        // Set width/height on group just for output the viewport size.
        return {
            root: root,
            width: width,
            height: height,
            viewBoxRect: viewBoxRect,
            viewBoxTransform: viewBoxTransform,
            named: named
        };
    }

    private _parseNode(
        xmlNode: SVGElement,
        parentGroup: Group,
        named: SVGParserResultNamedItem[],
        namedFrom: SVGParserResultNamedItem['namedFrom'],
        isInDefs: boolean,
        isInText: boolean
    ): void {

        const nodeName = xmlNode.nodeName.toLowerCase() as SVGNodeTagLower;

        // TODO:
        // support <style>...</style> in svg, where nodeName is 'style',
        // CSS classes is defined globally wherever the style tags are declared.

        let el;
        let namedFromForSub = namedFrom;

        if (nodeName === 'defs') {
            isInDefs = true;
        }
        if (nodeName === 'text') {
            isInText = true;
        }

        if (nodeName === 'defs' || nodeName === 'switch') {
            // Just make <switch> displayable. Do not support
            // the full feature of it.
            el = parentGroup;
        }
        else {
            // In <defs>, elments will not be rendered.
            // TODO:
            // do not support elements in <defs> yet, until requirement come.
            // other graphic elements can also be in <defs> and referenced by
            // <use x="5" y="5" xlink:href="#myCircle" />
            // multiple times
            if (!isInDefs) {
                const parser = nodeParsers[nodeName];
                if (parser && hasOwn(nodeParsers, nodeName)) {

                    el = parser.call(this, xmlNode, parentGroup);

                    // Do not support empty string;
                    const nameAttr = xmlNode.getAttribute('name');
                    if (nameAttr) {
                        const newNamed: SVGParserResultNamedItem = {
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

            // Whether gradients/patterns are declared in <defs> or not,
            // they all work.
            const parser = paintServerParsers[nodeName];
            if (parser && hasOwn(paintServerParsers, nodeName)) {
                const def = parser.call(this, xmlNode);
                const id = xmlNode.getAttribute('id');
                if (id) {
                    this._defs[id] = def;
                }
            }
        }

        // If xmlNode is <g>, <text>, <tspan>, <defs>, <switch>,
        // el will be a group, and traverse the children.
        if (el && el.isGroup) {
            let child = xmlNode.firstChild as SVGElement;
            while (child) {
                if (child.nodeType === 1) {
                    this._parseNode(child, el as Group, named, namedFromForSub, isInDefs, isInText);
                }
                // Is plain text rather than a tagged node.
                else if (child.nodeType === 3 && isInText) {
                    this._parseText(child, el as Group);
                }
                child = child.nextSibling as SVGElement;
            }
        }

    }

    private _parseText(xmlNode: SVGElement, parentGroup: Group): TSpan {
        const text = new TSpan({
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

        const textStyle = text.style as TextStyleOptionExtended;
        const fontSize = textStyle.fontSize;
        if (fontSize && fontSize < 9) {
            // PENDING
            textStyle.fontSize = 9;
            text.scaleX *= fontSize / 9;
            text.scaleY *= fontSize / 9;
        }

        const font = (textStyle.fontSize || textStyle.fontFamily) && [
            textStyle.fontStyle,
            textStyle.fontWeight,
            (textStyle.fontSize || 12) + 'px',
            // If font properties are defined, `fontFamily` should not be ignored.
            textStyle.fontFamily || 'sans-serif'
        ].join(' ');
        // Make font
        textStyle.font = font;

        const rect = text.getBoundingRect();
        this._textX += rect.width;

        parentGroup.add(text);

        return text;
    }

    static internalField = (function () {

        nodeParsers = {
            'g': function (xmlNode, parentGroup) {
                const g = new Group();
                inheritStyle(parentGroup, g);
                parseAttributes(xmlNode, g, this._defsUsePending, false, false);

                return g;
            },
            'rect': function (xmlNode, parentGroup) {
                const rect = new Rect();
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
                const circle = new Circle();
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
                const line = new Line();
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
                const ellipse = new Ellipse();
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
                const pointsStr = xmlNode.getAttribute('points');
                let pointsArr;
                if (pointsStr) {
                    pointsArr = parsePoints(pointsStr);
                }
                const polygon = new Polygon({
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
                const pointsStr = xmlNode.getAttribute('points');
                let pointsArr;
                if (pointsStr) {
                    pointsArr = parsePoints(pointsStr);
                }
                const polyline = new Polyline({
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
                const img = new ZRImage();
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
                const x = xmlNode.getAttribute('x') || '0';
                const y = xmlNode.getAttribute('y') || '0';
                const dx = xmlNode.getAttribute('dx') || '0';
                const dy = xmlNode.getAttribute('dy') || '0';

                this._textX = parseFloat(x) + parseFloat(dx);
                this._textY = parseFloat(y) + parseFloat(dy);

                const g = new Group();
                inheritStyle(parentGroup, g);
                parseAttributes(xmlNode, g, this._defsUsePending, false, true);

                return g;
            },
            'tspan': function (xmlNode, parentGroup) {
                const x = xmlNode.getAttribute('x');
                const y = xmlNode.getAttribute('y');
                if (x != null) {
                    // new offset x
                    this._textX = parseFloat(x);
                }
                if (y != null) {
                    // new offset y
                    this._textY = parseFloat(y);
                }
                const dx = xmlNode.getAttribute('dx') || '0';
                const dy = xmlNode.getAttribute('dy') || '0';

                const g = new Group();

                inheritStyle(parentGroup, g);
                parseAttributes(xmlNode, g, this._defsUsePending, false, true);

                this._textX += parseFloat(dx);
                this._textY += parseFloat(dy);

                return g;
            },
            'path': function (xmlNode, parentGroup) {
                // TODO svg fill rule
                // https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/fill-rule
                // path.style.globalCompositeOperation = 'xor';
                const d = xmlNode.getAttribute('d') || '';

                // Performance sensitive.

                const path = createFromString(d);

                inheritStyle(parentGroup, path);
                parseAttributes(xmlNode, path, this._defsUsePending, false, false);

                path.silent = true;

                return path;
            }
        };


    })();
}

const paintServerParsers: Dictionary<(xmlNode: SVGElement) => any> = {

    'lineargradient': function (xmlNode: SVGElement) {
        // TODO:
        // Support that x1,y1,x2,y2 are not declared lineargradient but in node.
        const x1 = parseInt(xmlNode.getAttribute('x1') || '0', 10);
        const y1 = parseInt(xmlNode.getAttribute('y1') || '0', 10);
        const x2 = parseInt(xmlNode.getAttribute('x2') || '10', 10);
        const y2 = parseInt(xmlNode.getAttribute('y2') || '0', 10);

        const gradient = new LinearGradient(x1, y1, x2, y2);

        parsePaintServerUnit(xmlNode, gradient);

        parseGradientColorStops(xmlNode, gradient);

        return gradient;
    },

    'radialgradient': function (xmlNode) {
        // TODO:
        // Support that x1,y1,x2,y2 are not declared radialgradient but in node.
        // TODO:
        // Support fx, fy, fr.
        const cx = parseInt(xmlNode.getAttribute('cx') || '0', 10);
        const cy = parseInt(xmlNode.getAttribute('cy') || '0', 10);
        const r = parseInt(xmlNode.getAttribute('r') || '0', 10);

        const gradient = new RadialGradient(cx, cy, r);

        parsePaintServerUnit(xmlNode, gradient);

        parseGradientColorStops(xmlNode, gradient);

        return gradient;
    }

    // TODO
    // 'pattern': function (xmlNode: SVGElement) {
    // }
};

function parsePaintServerUnit(xmlNode: SVGElement, gradient: Gradient) {
    const gradientUnits = xmlNode.getAttribute('gradientUnits');
    if (gradientUnits === 'userSpaceOnUse') {
        gradient.global = true;
    }
}

function parseGradientColorStops(xmlNode: SVGElement, gradient: GradientObject): void {

    let stop = xmlNode.firstChild as SVGStopElement;

    while (stop) {
        if (stop.nodeType === 1
            // there might be some other irrelevant tags used by editor.
            && stop.nodeName.toLocaleLowerCase() === 'stop'
        ) {
            const offsetStr = stop.getAttribute('offset');
            let offset: number;
            if (offsetStr && offsetStr.indexOf('%') > 0) {  // percentage
                offset = parseInt(offsetStr, 10) / 100;
            }
            else if (offsetStr) { // number from 0 to 1
                offset = parseFloat(offsetStr);
            }
            else {
                offset = 0;
            }

            // <stop style="stop-color:red"/> has higher priority than
            // <stop stop-color="red"/>
            const styleVals = {} as Dictionary<string>;
            parseInlineStyle(stop, styleVals, styleVals);
            const stopColor = styleVals.stopColor
                || stop.getAttribute('stop-color')
                || '#000000';

            gradient.colorStops.push({
                offset: offset,
                color: stopColor
            });
        }
        stop = stop.nextSibling as SVGStopElement;
    }
}

function inheritStyle(parent: Element, child: Element): void {
    if (parent && (parent as ElementExtended).__inheritedStyle) {
        if (!(child as ElementExtended).__inheritedStyle) {
            (child as ElementExtended).__inheritedStyle = {};
        }
        defaults((child as ElementExtended).__inheritedStyle, (parent as ElementExtended).__inheritedStyle);
    }
}

function parsePoints(pointsString: string): number[][] {
    const list = splitNumberSequence(pointsString);
    const points = [];

    for (let i = 0; i < list.length; i += 2) {
        const x = parseFloat(list[i]);
        const y = parseFloat(list[i + 1]);
        points.push([x, y]);
    }
    return points;
}

function parseAttributes(
    xmlNode: SVGElement,
    el: Element,
    defsUsePending: DefsUsePending,
    onlyInlineStyle: boolean,
    isTextGroup: boolean
): void {
    const disp = el as DisplayableExtended;
    const inheritedStyle = disp.__inheritedStyle = disp.__inheritedStyle || {};
    const selfStyle: SelfStyleByZRKey = {};

    // TODO Shadow
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
    ] as const, function (propName) {
        if (inheritedStyle[propName] != null) {
            disp.style[propName] = parseFloat(inheritedStyle[propName]);
        }
    });

    each([
        'lineDashOffset', 'lineCap', 'lineJoin', 'fontWeight', 'fontFamily', 'fontStyle', 'textAlign'
    ] as const, function (propName) {
        if (inheritedStyle[propName] != null) {
            disp.style[propName] = inheritedStyle[propName];
        }
    });

    // Because selfStyle only support textBaseline, so only text group need it.
    // in other cases selfStyle can be released.
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

function applyTextAlignment(
    text: TSpan,
    parentGroup: Group
): void {
    const parentSelfStyle = (parentGroup as ElementExtended).__selfStyle;
    if (parentSelfStyle) {
        const textBaseline = parentSelfStyle.textBaseline;
        let zrTextBaseline = textBaseline as CanvasTextBaseline;
        if (!textBaseline || textBaseline === 'auto') {
            // FIXME: 'auto' means the value is the dominant-baseline of the script to
            // which the character belongs - i.e., use the dominant-baseline of the parent.
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

    const parentInheritedStyle = (parentGroup as ElementExtended).__inheritedStyle;
    if (parentInheritedStyle) {
        // PENDING:
        // canvas `direction` is an experimental attribute.
        // so we do not support SVG direction "rtl" for text-anchor yet.
        const textAlign = parentInheritedStyle.textAlign;
        let zrTextAlign = textAlign as CanvasTextAlign;
        if (textAlign) {
            if (textAlign === 'middle') {
                zrTextAlign = 'center';
            }
            text.style.textAlign = zrTextAlign;
        }
    }
}

// Support `fill:url(#someId)`.
const urlRegex = /^url\(\s*#(.*?)\)/;
function getFillStrokeStyle(
    el: Displayable,
    method: 'fill' | 'stroke',
    str: string,
    defsUsePending: DefsUsePending
): string {
    const urlMatch = str && str.match(urlRegex);
    if (urlMatch) {
        const url = trim(urlMatch[1]);
        defsUsePending.push([el, method, url]);
        return;
    }
    // SVG fill and stroke can be 'none'.
    if (str === 'none') {
        str = null;
    }
    return str;
}

function applyDefs(
    defs: DefsMap,
    defsUsePending: DefsUsePending
): void {
    for (let i = 0; i < defsUsePending.length; i++) {
        const item = defsUsePending[i];
        item[0].style[item[1]] = defs[item[2]];
    }
}

// value can be like:
// '2e-4', 'l.5.9' (ignore 0), 'M-10-10', 'l-2.43e-1,34.9983',
// 'l-.5E1,54', '121-23-44-11' (no delimiter)
// PENDING: here continuous commas are treat as one comma, but the
// browser SVG parser treats this by printing error.
const numberReg = /-?([0-9]*\.)?[0-9]+([eE]-?[0-9]+)?/g;
function splitNumberSequence(rawStr: string): string[] {
    return rawStr.match(numberReg) || [];
}
// Most of the values can be separated by comma and/or white space.
// const DILIMITER_REG = /[\s,]+/;


const transformRegex = /(translate|scale|rotate|skewX|skewY|matrix)\(([\-\s0-9\.eE,]*)\)/g;
const DEGREE_TO_ANGLE = Math.PI / 180;

function parseTransformAttribute(xmlNode: SVGElement, node: Element): void {
    let transform = xmlNode.getAttribute('transform');
    if (transform) {
        transform = transform.replace(/,/g, ' ');
        const transformOps: string[] = [];
        let mt = null;
        transform.replace(transformRegex, function (str: string, type: string, value: string) {
            transformOps.push(type, value);
            return '';
        });

        for (let i = transformOps.length - 1; i > 0; i -= 2) {
            const value = transformOps[i];
            const type = transformOps[i - 1];
            const valueArr: string[] = splitNumberSequence(value);
            mt = mt || matrix.create();
            switch (type) {
                case 'translate':
                    matrix.translate(mt, mt, [parseFloat(valueArr[0]), parseFloat(valueArr[1] || '0')]);
                    break;
                case 'scale':
                    matrix.scale(mt, mt, [parseFloat(valueArr[0]), parseFloat(valueArr[1] || valueArr[0])]);
                    break;
                case 'rotate':
                    // TODO: zrender use different hand in coordinate system.
                    matrix.rotate(mt, mt, -parseFloat(valueArr[0]) * DEGREE_TO_ANGLE, [
                        parseFloat(valueArr[1] || '0'),
                        parseFloat(valueArr[2] || '0')
                    ]);
                    break;
                case 'skewX':
                    const sx = Math.tan(parseFloat(valueArr[0]) * DEGREE_TO_ANGLE);
                    matrix.mul(mt, [1, 0, sx, 1, 0, 0], mt);
                    break;
                case 'skewY':
                    const sy = Math.tan(parseFloat(valueArr[0]) * DEGREE_TO_ANGLE);
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

// Value may contain space.
const styleRegex = /([^\s:;]+)\s*:\s*([^:;]+)/g;
function parseInlineStyle(
    xmlNode: SVGElement,
    inheritableStyleResult: Dictionary<string>,
    selfStyleResult: Dictionary<string>
): void {
    const style = xmlNode.getAttribute('style');

    if (!style) {
        return;
    }

    styleRegex.lastIndex = 0;
    let styleRegResult;
    while ((styleRegResult = styleRegex.exec(style)) != null) {
        const svgStlAttr = styleRegResult[1];

        const zrInheritableStlAttr = hasOwn(INHERITABLE_STYLE_ATTRIBUTES_MAP, svgStlAttr)
            ? INHERITABLE_STYLE_ATTRIBUTES_MAP[svgStlAttr as keyof typeof INHERITABLE_STYLE_ATTRIBUTES_MAP]
            : null;
        if (zrInheritableStlAttr) {
            inheritableStyleResult[zrInheritableStlAttr] = styleRegResult[2];
        }

        const zrSelfStlAttr = hasOwn(SELF_STYLE_ATTRIBUTES_MAP, svgStlAttr)
            ? SELF_STYLE_ATTRIBUTES_MAP[svgStlAttr as keyof typeof SELF_STYLE_ATTRIBUTES_MAP]
            : null;
        if (zrSelfStlAttr) {
            selfStyleResult[zrSelfStlAttr] = styleRegResult[2];
        }
    }
}

function parseAttributeStyle(
    xmlNode: SVGElement,
    inheritableStyleResult: Dictionary<string>,
    selfStyleResult: Dictionary<string>
): void {
    for (let i = 0; i < INHERITABLE_STYLE_ATTRIBUTES_MAP_KEYS.length; i++) {
        const svgAttrName = INHERITABLE_STYLE_ATTRIBUTES_MAP_KEYS[i];
        const attrValue = xmlNode.getAttribute(svgAttrName);
        if (attrValue != null) {
            inheritableStyleResult[INHERITABLE_STYLE_ATTRIBUTES_MAP[svgAttrName]] = attrValue;
        }
    }
    for (let i = 0; i < SELF_STYLE_ATTRIBUTES_MAP_KEYS.length; i++) {
        const svgAttrName = SELF_STYLE_ATTRIBUTES_MAP_KEYS[i];
        const attrValue = xmlNode.getAttribute(svgAttrName);
        if (attrValue != null) {
            selfStyleResult[SELF_STYLE_ATTRIBUTES_MAP[svgAttrName]] = attrValue;
        }
    }
}

export function makeViewBoxTransform(viewBoxRect: RectLike, boundingRect: RectLike): {
    scale: number;
    x: number;
    y: number;
} {
    const scaleX = boundingRect.width / viewBoxRect.width;
    const scaleY = boundingRect.height / viewBoxRect.height;
    const scale = Math.min(scaleX, scaleY);
    // preserveAspectRatio 'xMidYMid'

    return {
        scale,
        x: -(viewBoxRect.x + viewBoxRect.width / 2) * scale + (boundingRect.x + boundingRect.width / 2),
        y: -(viewBoxRect.y + viewBoxRect.height / 2) * scale + (boundingRect.y + boundingRect.height / 2)
    };
}

export function parseSVG(xml: string | Document | SVGElement, opt: SVGParserOption): SVGParserResult {
    const parser = new SVGParser();
    return parser.parse(xml, opt);
}


// Also export parseXML to avoid breaking change.
export {parseXML};
