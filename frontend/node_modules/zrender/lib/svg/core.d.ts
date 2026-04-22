export declare type CSSSelectorVNode = Record<string, string>;
export declare type CSSAnimationVNode = Record<string, Record<string, string>>;
export declare const SVGNS = "http://www.w3.org/2000/svg";
export declare const XLINKNS = "http://www.w3.org/1999/xlink";
export declare const XMLNS = "http://www.w3.org/2000/xmlns/";
export declare const XML_NAMESPACE = "http://www.w3.org/XML/1998/namespace";
export declare const META_DATA_PREFIX = "ecmeta_";
export declare function createElement(name: string): SVGElement;
export declare type SVGVNodeAttrs = Record<string, string | number | undefined | boolean>;
export interface SVGVNode {
    tag: string;
    attrs: SVGVNodeAttrs;
    children?: SVGVNode[];
    text?: string;
    elm?: Node;
    key: string;
}
export declare function createVNode(tag: string, key: string, attrs?: SVGVNodeAttrs, children?: SVGVNode[], text?: string): SVGVNode;
export declare function vNodeToString(el: SVGVNode, opts?: {
    newline?: boolean;
}): string;
export declare function getCssString(selectorNodes: Record<string, CSSSelectorVNode>, animationNodes: Record<string, CSSAnimationVNode>, opts?: {
    newline?: boolean;
}): string;
export interface BrushScope {
    zrId: string;
    shadowCache: Record<string, string>;
    gradientCache: Record<string, string>;
    patternCache: Record<string, string>;
    clipPathCache: Record<string, string>;
    defs: Record<string, SVGVNode>;
    cssNodes: Record<string, CSSSelectorVNode>;
    cssAnims: Record<string, Record<string, Record<string, string>>>;
    cssStyleCache: Record<string, string>;
    cssAnimIdx: number;
    shadowIdx: number;
    gradientIdx: number;
    patternIdx: number;
    clipPathIdx: number;
    animation?: boolean;
    emphasis?: boolean;
    willUpdate?: boolean;
    compress?: boolean;
    ssr?: boolean;
}
export declare function createBrushScope(zrId: string): BrushScope;
export declare function createSVGVNode(width: number | string, height: number | string, children?: SVGVNode[], useViewBox?: boolean): SVGVNode;
