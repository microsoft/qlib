import { keys, map } from '../core/util';
import { encodeHTML } from '../core/dom';

export type CSSSelectorVNode = Record<string, string>
export type CSSAnimationVNode = Record<string, Record<string, string>>

export const SVGNS = 'http://www.w3.org/2000/svg';
export const XLINKNS = 'http://www.w3.org/1999/xlink';
export const XMLNS = 'http://www.w3.org/2000/xmlns/';
export const XML_NAMESPACE = 'http://www.w3.org/XML/1998/namespace';
export const META_DATA_PREFIX = 'ecmeta_';

export function createElement(name: string) {
    return document.createElementNS(SVGNS, name);
}

export type SVGVNodeAttrs = Record<string, string | number | undefined | boolean>
export interface SVGVNode {
    tag: string,
    attrs: SVGVNodeAttrs,
    children?: SVGVNode[],
    text?: string

    // For patching
    elm?: Node
    key: string
};
export function createVNode(
    tag: string,
    key: string,
    attrs?: SVGVNodeAttrs,
    children?: SVGVNode[],
    text?: string
): SVGVNode {
    return {
        tag,
        attrs: attrs || {},
        children,
        text,
        key
    };
}

function createElementOpen(name: string, attrs?: SVGVNodeAttrs) {
    const attrsStr: string[] = [];
    if (attrs) {
        // eslint-disable-next-line
        for (let key in attrs) {
            const val = attrs[key];
            let part = key;
            // Same with the logic in patch.
            if (val === false) {
                continue;
            }
            else if (val !== true && val != null) {
                part += `="${val}"`;
            }
            attrsStr.push(part);
        }
    }
    return `<${name} ${attrsStr.join(' ')}>`;
}

function createElementClose(name: string) {
    return `</${name}>`;
}

export function vNodeToString(el: SVGVNode, opts?: {
    newline?: boolean
}) {
    opts = opts || {};
    const S = opts.newline ? '\n' : '';
    function convertElToString(el: SVGVNode): string {
        const {children, tag, attrs, text} = el;
        return createElementOpen(tag, attrs)
            + (tag !== 'style' ? encodeHTML(text) : text || '')
            + (children ? `${S}${map(children, child => convertElToString(child)).join(S)}${S}` : '')
            + createElementClose(tag);
    }
    return convertElToString(el);
}

export function getCssString(
    selectorNodes: Record<string, CSSSelectorVNode>,
    animationNodes: Record<string, CSSAnimationVNode>,
    opts?: {
        newline?: boolean
    }
) {
    opts = opts || {};
    const S = opts.newline ? '\n' : '';
    const bracketBegin = ` {${S}`;
    const bracketEnd = `${S}}`;
    const selectors = map(keys(selectorNodes), className => {
        return className + bracketBegin + map(keys(selectorNodes[className]), attrName => {
            return `${attrName}:${selectorNodes[className][attrName]};`;
        }).join(S) + bracketEnd;
    }).join(S);
    const animations = map(keys(animationNodes), (animationName) => {
        return `@keyframes ${animationName}${bracketBegin}` + map(keys(animationNodes[animationName]), percent => {
            return percent + bracketBegin + map(keys(animationNodes[animationName][percent]), attrName => {
                let val = animationNodes[animationName][percent][attrName];
                // postprocess
                if (attrName === 'd') {
                    val = `path("${val}")`;
                }
                return `${attrName}:${val};`;
            }).join(S) + bracketEnd;
        }).join(S) + bracketEnd;
    }).join(S);

    if (!selectors && !animations) {
        return '';
    }

    return ['<![CDATA[', selectors, animations, ']]>'].join(S);
}


export interface BrushScope {
    zrId: string

    shadowCache: Record<string, string>
    gradientCache: Record<string, string>
    patternCache: Record<string, string>
    clipPathCache: Record<string, string>

    defs: Record<string, SVGVNode>

    cssNodes: Record<string, CSSSelectorVNode>
    cssAnims: Record<string, Record<string, Record<string, string>>>
    /**
     * Cache for css style string, mapping from style string to class name.
     */
    cssStyleCache: Record<string, string>

    cssAnimIdx: number

    shadowIdx: number
    gradientIdx: number
    patternIdx: number
    clipPathIdx: number
    // configs
    /**
     * If create animates nodes.
     */
    animation?: boolean,
    /**
     * If create emphasis styles.
     */
    emphasis?: boolean,

    /**
     * If will update. Some optimization for string generation can't be applied.
     */
    willUpdate?: boolean

    /**
     * If compress the output string.
     */
    compress?: boolean

    ssr?: boolean
}

export function createBrushScope(zrId: string): BrushScope {
    return {
        zrId,
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

export function createSVGVNode(
    width: number | string,
    height: number | string,
    children?: SVGVNode[],
    useViewBox?: boolean
) {
    return createVNode(
        'svg',
        'root',
        {
            'width': width,
            'height': height,
            'xmlns': SVGNS,
            'xmlns:xlink': XLINKNS,
            'version': '1.1',
            'baseProfile': 'full',
            'viewBox': useViewBox ? `0 0 ${width} ${height}` : false
        },
        children
    );
}