import {devicePixelRatio} from '../config';
import * as util from '../core/util';
import Layer, { LayerConfig } from './Layer';
import requestAnimationFrame from '../animation/requestAnimationFrame';
import env from '../core/env';
import Displayable from '../graphic/Displayable';
import { WXCanvasRenderingContext } from '../core/types';
import { GradientObject } from '../graphic/Gradient';
import { ImagePatternObject } from '../graphic/Pattern';
import Storage from '../Storage';
import { brush, BrushScope, brushSingle } from './graphic';
import { PainterBase } from '../PainterBase';
import BoundingRect from '../core/BoundingRect';
import { REDRAW_BIT } from '../graphic/constants';
import { getSize } from './helper';
import type IncrementalDisplayable from '../graphic/IncrementalDisplayable';

const HOVER_LAYER_ZLEVEL = 1e5;
const CANVAS_ZLEVEL = 314159;

const EL_AFTER_INCREMENTAL_INC = 0.01;
const INCREMENTAL_INC = 0.001;


function isLayerValid(layer: Layer) {
    if (!layer) {
        return false;
    }

    if (layer.__builtin__) {
        return true;
    }

    if (typeof (layer.resize) !== 'function'
        || typeof (layer.refresh) !== 'function'
    ) {
        return false;
    }

    return true;
}

function createRoot(width: number, height: number) {
    const domRoot = document.createElement('div');

    // domRoot.onselectstart = returnFalse; // Avoid page selected
    domRoot.style.cssText = [
        'position:relative',
        // IOS13 safari probably has a compositing bug (z order of the canvas and the consequent
        // dom does not act as expected) when some of the parent dom has
        // `-webkit-overflow-scrolling: touch;` and the webpage is longer than one screen and
        // the canvas is not at the top part of the page.
        // Check `https://bugs.webkit.org/show_bug.cgi?id=203681` for more details. We remove
        // this `overflow:hidden` to avoid the bug.
        // 'overflow:hidden',
        'width:' + width + 'px',
        'height:' + height + 'px',
        'padding:0',
        'margin:0',
        'border-width:0'
    ].join(';') + ';';

    return domRoot;
}

interface CanvasPainterOption {
    devicePixelRatio?: number
    width?: number | string  // Can be 10 / 10px / auto
    height?: number | string,
    useDirtyRect?: boolean
}

export default class CanvasPainter implements PainterBase {

    type = 'canvas'

    root: HTMLElement

    dpr: number

    storage: Storage

    private _singleCanvas: boolean

    private _opts: CanvasPainterOption

    private _zlevelList: number[] = []

    private _prevDisplayList: Displayable[] = []

    private _layers: {[key: number]: Layer} = {} // key is zlevel

    private _layerConfig: {[key: number]: LayerConfig} = {} // key is zlevel

    /**
     * zrender will do compositing when root is a canvas and have multiple zlevels.
     */
    private _needsManuallyCompositing = false

    private _width: number
    private _height: number

    private _domRoot: HTMLElement

    private _hoverlayer: Layer

    private _redrawId: number

    private _backgroundColor: string | GradientObject | ImagePatternObject


    constructor(root: HTMLElement, storage: Storage, opts: CanvasPainterOption, id: number) {

        this.type = 'canvas';

        // In node environment using node-canvas
        const singleCanvas = !root.nodeName // In node ?
            || root.nodeName.toUpperCase() === 'CANVAS';

        this._opts = opts = util.extend({}, opts || {}) as CanvasPainterOption;

        /**
         * @type {number}
         */
        this.dpr = opts.devicePixelRatio || devicePixelRatio;
        /**
         * @type {boolean}
         * @private
         */
        this._singleCanvas = singleCanvas;
        /**
         * 绘图容器
         * @type {HTMLElement}
         */
        this.root = root;

        const rootStyle = root.style;

        if (rootStyle) {
            // @ts-ignore
            util.disableUserSelect(root);
            root.innerHTML = '';
        }

        /**
         * @type {module:zrender/Storage}
         */
        this.storage = storage;

        const zlevelList: number[] = this._zlevelList;

        this._prevDisplayList = [];

        const layers = this._layers;

        if (!singleCanvas) {
            this._width = getSize(root, 0, opts);
            this._height = getSize(root, 1, opts);

            const domRoot = this._domRoot = createRoot(
                this._width, this._height
            );
            root.appendChild(domRoot);
        }
        else {
            const rootCanvas = root as HTMLCanvasElement;
            let width = rootCanvas.width;
            let height = rootCanvas.height;

            if (opts.width != null) {
                // TODO sting?
                width = opts.width as number;
            }
            if (opts.height != null) {
                // TODO sting?
                height = opts.height as number;
            }
            this.dpr = opts.devicePixelRatio || 1;

            // Use canvas width and height directly
            rootCanvas.width = width * this.dpr;
            rootCanvas.height = height * this.dpr;

            this._width = width;
            this._height = height;

            // Create layer if only one given canvas
            // Device can be specified to create a high dpi image.
            const mainLayer = new Layer(rootCanvas, this, this.dpr);
            mainLayer.__builtin__ = true;
            mainLayer.initContext();
            // FIXME Use canvas width and height
            // mainLayer.resize(width, height);
            layers[CANVAS_ZLEVEL] = mainLayer;
            mainLayer.zlevel = CANVAS_ZLEVEL;
            // Not use common zlevel.
            zlevelList.push(CANVAS_ZLEVEL);

            this._domRoot = root;
        }
    }


    getType() {
        return 'canvas';
    }

    /**
     * If painter use a single canvas
     */
    isSingleCanvas() {
        return this._singleCanvas;
    }

    getViewportRoot() {
        return this._domRoot;
    }

    getViewportRootOffset() {
        const viewportRoot = this.getViewportRoot();
        if (viewportRoot) {
            return {
                offsetLeft: viewportRoot.offsetLeft || 0,
                offsetTop: viewportRoot.offsetTop || 0
            };
        }
    }

    /**
     * 刷新
     * @param paintAll 强制绘制所有displayable
     */
    refresh(paintAll?: boolean) {
        const list = this.storage.getDisplayList(true);
        const prevList = this._prevDisplayList;

        const zlevelList = this._zlevelList;

        this._redrawId = Math.random();

        this._paintList(list, prevList, paintAll, this._redrawId);

        // Paint custum layers
        for (let i = 0; i < zlevelList.length; i++) {
            const z = zlevelList[i];
            const layer = this._layers[z];
            if (!layer.__builtin__ && layer.refresh) {
                const clearColor = i === 0 ? this._backgroundColor : null;
                layer.refresh(clearColor);
            }
        }

        if (this._opts.useDirtyRect) {
            this._prevDisplayList = list.slice();
        }

        return this;
    }


    refreshHover() {
        this._paintHoverList(this.storage.getDisplayList(false));
    }

    private _paintHoverList(list: Displayable[]) {
        let len = list.length;
        let hoverLayer = this._hoverlayer;
        hoverLayer && hoverLayer.clear();

        if (!len) {
            return;
        }

        const scope: BrushScope = {
            inHover: true,
            viewWidth: this._width,
            viewHeight: this._height
        };

        let ctx;
        for (let i = 0; i < len; i++) {
            const el = list[i];
            if (el.__inHover) {
                // Use a extream large zlevel
                // FIXME?
                if (!hoverLayer) {
                    hoverLayer = this._hoverlayer = this.getLayer(HOVER_LAYER_ZLEVEL);
                }

                if (!ctx) {
                    ctx = hoverLayer.ctx;
                    ctx.save();
                }

                brush(ctx, el, scope, i === len - 1);
            }
        }
        if (ctx) {
            ctx.restore();
        }
    }

    getHoverLayer() {
        return this.getLayer(HOVER_LAYER_ZLEVEL);
    }

    paintOne(ctx: CanvasRenderingContext2D, el: Displayable) {
        brushSingle(ctx, el);
    }

    private _paintList(list: Displayable[], prevList: Displayable[], paintAll: boolean, redrawId?: number) {
        if (this._redrawId !== redrawId) {
            return;
        }

        paintAll = paintAll || false;

        this._updateLayerStatus(list);

        const {finished, needsRefreshHover} = this._doPaintList(list, prevList, paintAll);

        if (this._needsManuallyCompositing) {
            this._compositeManually();
        }

        if (needsRefreshHover) {
            this._paintHoverList(list);
        }

        if (!finished) {
            const self = this;
            requestAnimationFrame(function () {
                self._paintList(list, prevList, paintAll, redrawId);
            });
        }
        else {
            this.eachLayer(layer => {
                layer.afterBrush && layer.afterBrush();
            });
        }
    }

    private _compositeManually() {
        const ctx = this.getLayer(CANVAS_ZLEVEL).ctx;
        const width = (this._domRoot as HTMLCanvasElement).width;
        const height = (this._domRoot as HTMLCanvasElement).height;
        ctx.clearRect(0, 0, width, height);
        // PENDING, If only builtin layer?
        this.eachBuiltinLayer(function (layer) {
            if (layer.virtual) {
                ctx.drawImage(layer.dom, 0, 0, width, height);
            }
        });
    }

    private _doPaintList(
        list: Displayable[],
        prevList: Displayable[],
        paintAll?: boolean
    ): {
        finished: boolean
        needsRefreshHover: boolean
    } {
        const layerList = [];
        const useDirtyRect = this._opts.useDirtyRect;
        for (let zi = 0; zi < this._zlevelList.length; zi++) {
            const zlevel = this._zlevelList[zi];
            const layer = this._layers[zlevel];
            if (layer.__builtin__
                && layer !== this._hoverlayer
                && (layer.__dirty || paintAll)
                // Layer with hover elements can't be redrawn.
                // && !layer.__hasHoverLayerELement
            ) {
                layerList.push(layer);
            }
        }

        let finished = true;
        let needsRefreshHover = false;

        for (let k = 0; k < layerList.length; k++) {
            const layer = layerList[k];
            const ctx = layer.ctx;

            const repaintRects = useDirtyRect
                && layer.createRepaintRects(list, prevList, this._width, this._height);

            let start = paintAll ? layer.__startIndex : layer.__drawIndex;

            const useTimer = !paintAll && layer.incremental && Date.now;
            const startTime = useTimer && Date.now();

            const clearColor = layer.zlevel === this._zlevelList[0]
                ? this._backgroundColor : null;

            // All elements in this layer are removed.
            if (layer.__startIndex === layer.__endIndex) {
                layer.clear(false, clearColor, repaintRects);
            }
            else if (start === layer.__startIndex) {
                const firstEl = list[start];
                if (!firstEl.incremental || !(firstEl as IncrementalDisplayable).notClear || paintAll) {
                    layer.clear(false, clearColor, repaintRects);
                }
            }
            if (start === -1) {
                console.error('For some unknown reason. drawIndex is -1');
                start = layer.__startIndex;
            }
            let i: number;
            /* eslint-disable-next-line */
            const repaint = (repaintRect?: BoundingRect) => {
                const scope: BrushScope = {
                    inHover: false,
                    allClipped: false,
                    prevEl: null,
                    viewWidth: this._width,
                    viewHeight: this._height
                };

                for (i = start; i < layer.__endIndex; i++) {
                    const el = list[i];

                    if (el.__inHover) {
                        needsRefreshHover = true;
                    }

                    this._doPaintEl(el, layer, useDirtyRect, repaintRect, scope, i === layer.__endIndex - 1);

                    if (useTimer) {
                        // Date.now can be executed in 13,025,305 ops/second.
                        const dTime = Date.now() - startTime;
                        // Give 15 millisecond to draw.
                        // The rest elements will be drawn in the next frame.
                        if (dTime > 15) {
                            break;
                        }
                    }
                }

                if (scope.prevElClipPaths) {
                    // Needs restore the state. If last drawn element is in the clipping area.
                    ctx.restore();
                }
            };

            if (repaintRects) {
                if (repaintRects.length === 0) {
                    // Nothing to repaint, mark as finished
                    i = layer.__endIndex;
                }
                else {
                    const dpr = this.dpr;
                    // Set repaintRect as clipPath
                    for (var r = 0; r < repaintRects.length; ++r) {
                        const rect = repaintRects[r];

                        ctx.save();
                        ctx.beginPath();
                        ctx.rect(
                            rect.x * dpr,
                            rect.y * dpr,
                            rect.width * dpr,
                            rect.height * dpr
                        );
                        ctx.clip();

                        repaint(rect);
                        ctx.restore();
                    }
                }
            }
            else {
                // Paint all once
                ctx.save();
                repaint();
                ctx.restore();
            }

            layer.__drawIndex = i;

            if (layer.__drawIndex < layer.__endIndex) {
                finished = false;
            }
        }

        if (env.wxa) {
            // Flush for weixin application
            util.each(this._layers, function (layer) {
                if (layer && layer.ctx && (layer.ctx as WXCanvasRenderingContext).draw) {
                    (layer.ctx as WXCanvasRenderingContext).draw();
                }
            });
        }

        return {
            finished,
            needsRefreshHover
        };
    }

    private _doPaintEl(
        el: Displayable,
        currentLayer: Layer,
        useDirtyRect: boolean,
        repaintRect: BoundingRect,
        scope: BrushScope,
        isLast: boolean
    ) {
        const ctx = currentLayer.ctx;
        if (useDirtyRect) {
            const paintRect = el.getPaintRect();
            if (!repaintRect || paintRect && paintRect.intersect(repaintRect)) {
                brush(ctx, el, scope, isLast);
                el.setPrevPaintRect(paintRect);
            }
        }
        else {
            brush(ctx, el, scope, isLast);
        }
    }

    /**
     * 获取 zlevel 所在层，如果不存在则会创建一个新的层
     * @param zlevel
     * @param virtual Virtual layer will not be inserted into dom.
     */
    getLayer(zlevel: number, virtual?: boolean) {
        if (this._singleCanvas && !this._needsManuallyCompositing) {
            zlevel = CANVAS_ZLEVEL;
        }
        let layer = this._layers[zlevel];
        if (!layer) {
            // Create a new layer
            layer = new Layer('zr_' + zlevel, this, this.dpr);
            layer.zlevel = zlevel;
            layer.__builtin__ = true;

            if (this._layerConfig[zlevel]) {
                util.merge(layer, this._layerConfig[zlevel], true);
            }
            // TODO Remove EL_AFTER_INCREMENTAL_INC magic number
            else if (this._layerConfig[zlevel - EL_AFTER_INCREMENTAL_INC]) {
                util.merge(layer, this._layerConfig[zlevel - EL_AFTER_INCREMENTAL_INC], true);
            }

            if (virtual) {
                layer.virtual = virtual;
            }

            this.insertLayer(zlevel, layer);

            // Context is created after dom inserted to document
            // Or excanvas will get 0px clientWidth and clientHeight
            layer.initContext();
        }

        return layer;
    }

    insertLayer(zlevel: number, layer: Layer) {

        const layersMap = this._layers;
        const zlevelList = this._zlevelList;
        const len = zlevelList.length;
        const domRoot = this._domRoot;
        let prevLayer = null;
        let i = -1;

        if (layersMap[zlevel]) {
            if (process.env.NODE_ENV !== 'production') {
                util.logError('ZLevel ' + zlevel + ' has been used already');
            }
            return;
        }
        // Check if is a valid layer
        if (!isLayerValid(layer)) {
            if (process.env.NODE_ENV !== 'production') {
                util.logError('Layer of zlevel ' + zlevel + ' is not valid');
            }
            return;
        }

        if (len > 0 && zlevel > zlevelList[0]) {
            for (i = 0; i < len - 1; i++) {
                if (
                    zlevelList[i] < zlevel
                    && zlevelList[i + 1] > zlevel
                ) {
                    break;
                }
            }
            prevLayer = layersMap[zlevelList[i]];
        }
        zlevelList.splice(i + 1, 0, zlevel);

        layersMap[zlevel] = layer;

        // Virtual layer will not directly show on the screen.
        // (It can be a WebGL layer and assigned to a ZRImage element)
        // But it still under management of zrender.
        if (!layer.virtual) {
            if (prevLayer) {
                const prevDom = prevLayer.dom;
                if (prevDom.nextSibling) {
                    domRoot.insertBefore(
                        layer.dom,
                        prevDom.nextSibling
                    );
                }
                else {
                    domRoot.appendChild(layer.dom);
                }
            }
            else {
                if (domRoot.firstChild) {
                    domRoot.insertBefore(layer.dom, domRoot.firstChild);
                }
                else {
                    domRoot.appendChild(layer.dom);
                }
            }
        }

        layer.painter || (layer.painter = this);
    }

    // Iterate each layer
    eachLayer<T>(cb: (this: T, layer: Layer, z: number) => void, context?: T) {
        const zlevelList = this._zlevelList;
        for (let i = 0; i < zlevelList.length; i++) {
            const z = zlevelList[i];
            cb.call(context, this._layers[z], z);
        }
    }

    // Iterate each buildin layer
    eachBuiltinLayer<T>(cb: (this: T, layer: Layer, z: number) => void, context?: T) {
        const zlevelList = this._zlevelList;
        for (let i = 0; i < zlevelList.length; i++) {
            const z = zlevelList[i];
            const layer = this._layers[z];
            if (layer.__builtin__) {
                cb.call(context, layer, z);
            }
        }
    }

    // Iterate each other layer except buildin layer
    eachOtherLayer<T>(cb: (this: T, layer: Layer, z: number) => void, context?: T) {
        const zlevelList = this._zlevelList;
        for (let i = 0; i < zlevelList.length; i++) {
            const z = zlevelList[i];
            const layer = this._layers[z];
            if (!layer.__builtin__) {
                cb.call(context, layer, z);
            }
        }
    }

    /**
     * 获取所有已创建的层
     * @param prevLayer
     */
    getLayers() {
        return this._layers;
    }

    _updateLayerStatus(list: Displayable[]) {

        this.eachBuiltinLayer(function (layer, z) {
            layer.__dirty = layer.__used = false;
        });

        function updatePrevLayer(idx: number) {
            if (prevLayer) {
                if (prevLayer.__endIndex !== idx) {
                    prevLayer.__dirty = true;
                }
                prevLayer.__endIndex = idx;
            }
        }

        if (this._singleCanvas) {
            for (let i = 1; i < list.length; i++) {
                const el = list[i];
                if (el.zlevel !== list[i - 1].zlevel || el.incremental) {
                    this._needsManuallyCompositing = true;
                    break;
                }
            }
        }

        let prevLayer: Layer = null;
        let incrementalLayerCount = 0;
        let prevZlevel;
        let i;

        for (i = 0; i < list.length; i++) {
            const el = list[i];
            const zlevel = el.zlevel;
            let layer;

            if (prevZlevel !== zlevel) {
                prevZlevel = zlevel;
                incrementalLayerCount = 0;
            }

            // TODO Not use magic number on zlevel.

            // Each layer with increment element can be separated to 3 layers.
            //          (Other Element drawn after incremental element)
            // -----------------zlevel + EL_AFTER_INCREMENTAL_INC--------------------
            //                      (Incremental element)
            // ----------------------zlevel + INCREMENTAL_INC------------------------
            //              (Element drawn before incremental element)
            // --------------------------------zlevel--------------------------------
            if (el.incremental) {
                layer = this.getLayer(zlevel + INCREMENTAL_INC, this._needsManuallyCompositing);
                layer.incremental = true;
                incrementalLayerCount = 1;
            }
            else {
                layer = this.getLayer(
                    zlevel + (incrementalLayerCount > 0 ? EL_AFTER_INCREMENTAL_INC : 0),
                    this._needsManuallyCompositing
                );
            }

            if (!layer.__builtin__) {
                util.logError('ZLevel ' + zlevel + ' has been used by unkown layer ' + layer.id);
            }

            if (layer !== prevLayer) {
                layer.__used = true;
                if (layer.__startIndex !== i) {
                    layer.__dirty = true;
                }
                layer.__startIndex = i;
                if (!layer.incremental) {
                    layer.__drawIndex = i;
                }
                else {
                    // Mark layer draw index needs to update.
                    layer.__drawIndex = -1;
                }
                updatePrevLayer(i);
                prevLayer = layer;
            }
            if ((el.__dirty & REDRAW_BIT) && !el.__inHover) {  // Ignore dirty elements in hover layer.
                layer.__dirty = true;
                if (layer.incremental && layer.__drawIndex < 0) {
                    // Start draw from the first dirty element.
                    layer.__drawIndex = i;
                }
            }
        }

        updatePrevLayer(i);

        this.eachBuiltinLayer(function (layer, z) {
            // Used in last frame but not in this frame. Needs clear
            if (!layer.__used && layer.getElementCount() > 0) {
                layer.__dirty = true;
                layer.__startIndex = layer.__endIndex = layer.__drawIndex = 0;
            }
            // For incremental layer. In case start index changed and no elements are dirty.
            if (layer.__dirty && layer.__drawIndex < 0) {
                layer.__drawIndex = layer.__startIndex;
            }
        });
    }

    /**
     * 清除hover层外所有内容
     */
    clear() {
        this.eachBuiltinLayer(this._clearLayer);
        return this;
    }

    _clearLayer(layer: Layer) {
        layer.clear();
    }

    setBackgroundColor(backgroundColor: string | GradientObject | ImagePatternObject) {
        this._backgroundColor = backgroundColor;

        util.each(this._layers, layer => {
            layer.setUnpainted();
        });
    }

    /**
     * 修改指定zlevel的绘制参数
     */
    configLayer(zlevel: number, config: LayerConfig) {
        if (config) {
            const layerConfig = this._layerConfig;
            if (!layerConfig[zlevel]) {
                layerConfig[zlevel] = config;
            }
            else {
                util.merge(layerConfig[zlevel], config, true);
            }

            for (let i = 0; i < this._zlevelList.length; i++) {
                const _zlevel = this._zlevelList[i];
                // TODO Remove EL_AFTER_INCREMENTAL_INC magic number
                if (_zlevel === zlevel || _zlevel === zlevel + EL_AFTER_INCREMENTAL_INC) {
                    const layer = this._layers[_zlevel];
                    util.merge(layer, layerConfig[zlevel], true);
                }
            }
        }
    }

    /**
     * 删除指定层
     * @param zlevel 层所在的zlevel
     */
    delLayer(zlevel: number) {
        const layers = this._layers;
        const zlevelList = this._zlevelList;
        const layer = layers[zlevel];
        if (!layer) {
            return;
        }
        layer.dom.parentNode.removeChild(layer.dom);
        delete layers[zlevel];

        zlevelList.splice(util.indexOf(zlevelList, zlevel), 1);
    }

    /**
     * 区域大小变化后重绘
     */
    resize(
        width?: number | string,
        height?: number | string
    ) {
        if (!this._domRoot.style) { // Maybe in node or worker
            if (width == null || height == null) {
                return;
            }
            // TODO width / height may be string
            this._width = width as number;
            this._height = height as number;

            this.getLayer(CANVAS_ZLEVEL).resize(width as number, height as number);
        }
        else {
            const domRoot = this._domRoot;
            // FIXME Why ?
            domRoot.style.display = 'none';

            // Save input w/h
            const opts = this._opts;
            const root = this.root;
            width != null && (opts.width = width);
            height != null && (opts.height = height);

            width = getSize(root, 0, opts);
            height = getSize(root, 1, opts);

            domRoot.style.display = '';

            // 优化没有实际改变的resize
            if (this._width !== width || height !== this._height) {
                domRoot.style.width = width + 'px';
                domRoot.style.height = height + 'px';

                for (let id in this._layers) {
                    if (this._layers.hasOwnProperty(id)) {
                        this._layers[id].resize(width, height);
                    }
                }

                this.refresh(true);
            }

            this._width = width;
            this._height = height;

        }
        return this;
    }

    /**
     * 清除单独的一个层
     * @param {number} zlevel
     */
    clearLayer(zlevel: number) {
        const layer = this._layers[zlevel];
        if (layer) {
            layer.clear();
        }
    }

    /**
     * 释放
     */
    dispose() {
        this.root.innerHTML = '';

        this.root =
        this.storage =

        this._domRoot =
        this._layers = null;
    }

    /**
     * Get canvas which has all thing rendered
     */
    getRenderedCanvas(opts?: {
        backgroundColor?: string | GradientObject | ImagePatternObject
        pixelRatio?: number
    }) {
        opts = opts || {};
        if (this._singleCanvas && !this._compositeManually) {
            return this._layers[CANVAS_ZLEVEL].dom;
        }

        const imageLayer = new Layer('image', this, opts.pixelRatio || this.dpr);
        imageLayer.initContext();
        imageLayer.clear(false, opts.backgroundColor || this._backgroundColor);

        const ctx = imageLayer.ctx;

        if (opts.pixelRatio <= this.dpr) {
            this.refresh();

            const width = imageLayer.dom.width;
            const height = imageLayer.dom.height;
            this.eachLayer(function (layer) {
                if (layer.__builtin__) {
                    ctx.drawImage(layer.dom, 0, 0, width, height);
                }
                else if (layer.renderToCanvas) {
                    ctx.save();
                    layer.renderToCanvas(ctx);
                    ctx.restore();
                }
            });
        }
        else {
            // PENDING, echarts-gl and incremental rendering.
            const scope = {
                inHover: false,
                viewWidth: this._width,
                viewHeight: this._height
            };
            const displayList = this.storage.getDisplayList(true);
            for (let i = 0, len = displayList.length; i < len; i++) {
                const el = displayList[i];
                brush(ctx, el, scope, i === len - 1);
            }
        }

        return imageLayer.dom;
    }
    /**
     * 获取绘图区域宽度
     */
    getWidth() {
        return this._width;
    }

    /**
     * 获取绘图区域高度
     */
    getHeight() {
        return this._height;
    }
};