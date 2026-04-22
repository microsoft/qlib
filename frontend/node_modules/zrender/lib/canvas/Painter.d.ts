import Layer, { LayerConfig } from './Layer';
import Displayable from '../graphic/Displayable';
import { GradientObject } from '../graphic/Gradient';
import { ImagePatternObject } from '../graphic/Pattern';
import Storage from '../Storage';
import { PainterBase } from '../PainterBase';
interface CanvasPainterOption {
    devicePixelRatio?: number;
    width?: number | string;
    height?: number | string;
    useDirtyRect?: boolean;
}
export default class CanvasPainter implements PainterBase {
    type: string;
    root: HTMLElement;
    dpr: number;
    storage: Storage;
    private _singleCanvas;
    private _opts;
    private _zlevelList;
    private _prevDisplayList;
    private _layers;
    private _layerConfig;
    private _needsManuallyCompositing;
    private _width;
    private _height;
    private _domRoot;
    private _hoverlayer;
    private _redrawId;
    private _backgroundColor;
    constructor(root: HTMLElement, storage: Storage, opts: CanvasPainterOption, id: number);
    getType(): string;
    isSingleCanvas(): boolean;
    getViewportRoot(): HTMLElement;
    getViewportRootOffset(): {
        offsetLeft: number;
        offsetTop: number;
    };
    refresh(paintAll?: boolean): this;
    refreshHover(): void;
    private _paintHoverList;
    getHoverLayer(): Layer;
    paintOne(ctx: CanvasRenderingContext2D, el: Displayable): void;
    private _paintList;
    private _compositeManually;
    private _doPaintList;
    private _doPaintEl;
    getLayer(zlevel: number, virtual?: boolean): Layer;
    insertLayer(zlevel: number, layer: Layer): void;
    eachLayer<T>(cb: (this: T, layer: Layer, z: number) => void, context?: T): void;
    eachBuiltinLayer<T>(cb: (this: T, layer: Layer, z: number) => void, context?: T): void;
    eachOtherLayer<T>(cb: (this: T, layer: Layer, z: number) => void, context?: T): void;
    getLayers(): {
        [key: number]: Layer;
    };
    _updateLayerStatus(list: Displayable[]): void;
    clear(): this;
    _clearLayer(layer: Layer): void;
    setBackgroundColor(backgroundColor: string | GradientObject | ImagePatternObject): void;
    configLayer(zlevel: number, config: LayerConfig): void;
    delLayer(zlevel: number): void;
    resize(width?: number | string, height?: number | string): this;
    clearLayer(zlevel: number): void;
    dispose(): void;
    getRenderedCanvas(opts?: {
        backgroundColor?: string | GradientObject | ImagePatternObject;
        pixelRatio?: number;
    }): HTMLCanvasElement;
    getWidth(): number;
    getHeight(): number;
}
export {};
