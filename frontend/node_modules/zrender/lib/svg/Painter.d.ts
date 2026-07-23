import Displayable from '../graphic/Displayable';
import Storage from '../Storage';
import { PainterBase } from '../PainterBase';
import { SVGVNode, BrushScope } from './core';
import { GradientObject } from '../graphic/Gradient';
import { PatternObject } from '../graphic/Pattern';
interface SVGPainterOption {
    width?: number;
    height?: number;
    ssr?: boolean;
}
declare type SVGPainterBackgroundColor = string | GradientObject | PatternObject;
declare class SVGPainter implements PainterBase {
    type: string;
    storage: Storage;
    root: HTMLElement;
    private _svgDom;
    private _viewport;
    private _opts;
    private _oldVNode;
    private _bgVNode;
    private _mainVNode;
    private _width;
    private _height;
    private _backgroundColor;
    private _id;
    constructor(root: HTMLElement, storage: Storage, opts: SVGPainterOption);
    getType(): string;
    getViewportRoot(): HTMLElement;
    getViewportRootOffset(): {
        offsetLeft: number;
        offsetTop: number;
    };
    getSvgDom(): SVGElement;
    refresh(): void;
    renderOneToVNode(el: Displayable): SVGVNode;
    renderToVNode(opts?: {
        animation?: boolean;
        willUpdate?: boolean;
        compress?: boolean;
        useViewBox?: boolean;
        emphasis?: boolean;
    }): SVGVNode;
    renderToString(opts?: {
        cssAnimation?: boolean;
        cssEmphasis?: boolean;
        useViewBox?: boolean;
    }): string;
    setBackgroundColor(backgroundColor: SVGPainterBackgroundColor): void;
    getSvgRoot(): SVGElement;
    _paintList(list: Displayable[], scope: BrushScope, out?: SVGVNode[]): void;
    resize(width: number, height: number): void;
    getWidth(): number;
    getHeight(): number;
    dispose(): void;
    clear(): void;
    toDataURL(base64?: boolean): string;
    refreshHover: () => void;
    configLayer: (zlevel: number, config: import("../core/types").Dictionary<any>) => void;
}
export default SVGPainter;
