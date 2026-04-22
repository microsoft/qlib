import * as util from '../core/util';
import {devicePixelRatio} from '../config';
import { ImagePatternObject } from '../graphic/Pattern';
import CanvasPainter from './Painter';
import { GradientObject, InnerGradientObject } from '../graphic/Gradient';
import { ZRCanvasRenderingContext } from '../core/types';
import Eventful from '../core/Eventful';
import { ElementEventCallback } from '../Element';
import { getCanvasGradient } from './helper';
import { createCanvasPattern } from './graphic';
import Displayable from '../graphic/Displayable';
import BoundingRect from '../core/BoundingRect';
import { REDRAW_BIT } from '../graphic/constants';
import { platformApi } from '../core/platform';

function createDom(id: string, painter: CanvasPainter, dpr: number) {
    const newDom = platformApi.createCanvas();
    const width = painter.getWidth();
    const height = painter.getHeight();

    const newDomStyle = newDom.style;
    if (newDomStyle) {  // In node or some other non-browser environment
        newDomStyle.position = 'absolute';
        newDomStyle.left = '0';
        newDomStyle.top = '0';
        newDomStyle.width = width + 'px';
        newDomStyle.height = height + 'px';

        newDom.setAttribute('data-zr-dom-id', id);
    }

    newDom.width = width * dpr;
    newDom.height = height * dpr;

    return newDom;
}

export interface LayerConfig {
    // 每次清空画布的颜色
    clearColor?: string | GradientObject | ImagePatternObject
    // 是否开启动态模糊
    motionBlur?: boolean
    // 在开启动态模糊的时候使用，与上一帧混合的alpha值，值越大尾迹越明显
    lastFrameAlpha?: number
};

export default class Layer extends Eventful {

    id: string

    dom: HTMLCanvasElement
    domBack: HTMLCanvasElement

    ctx: CanvasRenderingContext2D
    ctxBack: CanvasRenderingContext2D

    painter: CanvasPainter

    // Configs
    /**
     * 每次清空画布的颜色
     */
    clearColor: string | GradientObject | ImagePatternObject
    /**
     * 是否开启动态模糊
     */
    motionBlur = false
    /**
     * 在开启动态模糊的时候使用，与上一帧混合的alpha值，值越大尾迹越明显
     */
    lastFrameAlpha = 0.7
    /**
     * Layer dpr
     */
    dpr = 1

    /**
     * Virtual layer will not be inserted into dom.
     */
    virtual = false

    config = {}

    incremental = false

    zlevel = 0

    maxRepaintRectCount = 5

    private _paintRects: BoundingRect[]

    __dirty = true
    __firstTimePaint = true

    __used = false

    __drawIndex = 0
    __startIndex = 0
    __endIndex = 0

    // indices in the previous frame
    __prevStartIndex: number = null
    __prevEndIndex: number = null

    __builtin__: boolean

    constructor(id: string | HTMLCanvasElement, painter: CanvasPainter, dpr?: number) {
        super();

        let dom;
        dpr = dpr || devicePixelRatio;
        if (typeof id === 'string') {
            dom = createDom(id, painter, dpr);
        }
        // Not using isDom because in node it will return false
        else if (util.isObject(id)) {
            dom = id;
            id = dom.id;
        }
        this.id = id as string;
        this.dom = dom;

        const domStyle = dom.style;
        if (domStyle) { // Not in node
            util.disableUserSelect(dom);
            dom.onselectstart = () => false;
            domStyle.padding = '0';
            domStyle.margin = '0';
            domStyle.borderWidth = '0';
        }

        this.painter = painter;

        this.dpr = dpr;
    }

    getElementCount() {
        return this.__endIndex - this.__startIndex;
    }

    afterBrush() {
        this.__prevStartIndex = this.__startIndex;
        this.__prevEndIndex = this.__endIndex;
    }

    initContext() {
        this.ctx = this.dom.getContext('2d');
        (this.ctx as ZRCanvasRenderingContext).dpr = this.dpr;
    }

    setUnpainted() {
        this.__firstTimePaint = true;
    }

    createBackBuffer() {
        const dpr = this.dpr;

        this.domBack = createDom('back-' + this.id, this.painter, dpr);
        this.ctxBack = this.domBack.getContext('2d');

        if (dpr !== 1) {
            this.ctxBack.scale(dpr, dpr);
        }
    }

    /**
     * Create repaint list when using dirty rect rendering.
     *
     * @param displayList current rendering list
     * @param prevList last frame rendering list
     * @return repaint rects. null for the first frame, [] for no element dirty
     */
    createRepaintRects(
        displayList: Displayable[],
        prevList: Displayable[],
        viewWidth: number,
        viewHeight: number
    ) {
        if (this.__firstTimePaint) {
            this.__firstTimePaint = false;
            return null;
        }

        const mergedRepaintRects: BoundingRect[] = [];
        const maxRepaintRectCount = this.maxRepaintRectCount;
        let full = false;
        const pendingRect = new BoundingRect(0, 0, 0, 0);

        function addRectToMergePool(rect: BoundingRect) {
            if (!rect.isFinite() || rect.isZero()) {
                return;
            }

            if (mergedRepaintRects.length === 0) {
                // First rect, create new merged rect
                const boundingRect = new BoundingRect(0, 0, 0, 0);
                boundingRect.copy(rect);
                mergedRepaintRects.push(boundingRect);
            }
            else {
                let isMerged = false;
                let minDeltaArea = Infinity;
                let bestRectToMergeIdx = 0;
                for (let i = 0; i < mergedRepaintRects.length; ++i) {
                    const mergedRect = mergedRepaintRects[i];

                    // Merge if has intersection
                    if (mergedRect.intersect(rect)) {
                        const pendingRect = new BoundingRect(0, 0, 0, 0);
                        pendingRect.copy(mergedRect);
                        pendingRect.union(rect);
                        mergedRepaintRects[i] = pendingRect;
                        isMerged = true;
                        break;
                    }
                    else if (full) {
                        // Merged to exists rectangles if full
                        pendingRect.copy(rect);
                        pendingRect.union(mergedRect);
                        const aArea = rect.width * rect.height;
                        const bArea = mergedRect.width * mergedRect.height;
                        const pendingArea = pendingRect.width * pendingRect.height;
                        const deltaArea = pendingArea - aArea - bArea;
                        if (deltaArea < minDeltaArea) {
                            minDeltaArea = deltaArea;
                            bestRectToMergeIdx = i;
                        }
                    }
                }

                if (full) {
                    mergedRepaintRects[bestRectToMergeIdx].union(rect);
                    isMerged = true;
                }

                if (!isMerged) {
                    // Create new merged rect if cannot merge with current
                    const boundingRect = new BoundingRect(0, 0, 0, 0);
                    boundingRect.copy(rect);
                    mergedRepaintRects.push(boundingRect);
                }
                if (!full) {
                    full = mergedRepaintRects.length >= maxRepaintRectCount;
                }
            }
        }

        /**
         * Loop the paint list of this frame and get the dirty rects of elements
         * in this frame.
         */
        for (let i = this.__startIndex; i < this.__endIndex; ++i) {
            const el = displayList[i];
            if (el) {
                /**
                 * `shouldPaint` is true only when the element is not ignored or
                 * invisible and all its ancestors are not ignored.
                 * `shouldPaint` being true means it will be brushed this frame.
                 *
                 * `__isRendered` being true means the element is currently on
                 * the canvas.
                 *
                 * `__dirty` being true means the element should be brushed this
                 * frame.
                 *
                 * We only need to repaint the element's previous painting rect
                 * if it's currently on the canvas and needs repaint this frame
                 * or not painted this frame.
                 */
                const shouldPaint = el.shouldBePainted(viewWidth, viewHeight, true, true);
                const prevRect = el.__isRendered && ((el.__dirty & REDRAW_BIT) || !shouldPaint)
                    ? el.getPrevPaintRect()
                    : null;
                if (prevRect) {
                    addRectToMergePool(prevRect);
                }

                /**
                 * On the other hand, we only need to paint the current rect
                 * if the element should be brushed this frame and either being
                 * dirty or not rendered before.
                 */
                const curRect = shouldPaint && ((el.__dirty & REDRAW_BIT) || !el.__isRendered)
                    ? el.getPaintRect()
                    : null;
                if (curRect) {
                    addRectToMergePool(curRect);
                }
            }
        }

        /**
         * The above loop calculates the dirty rects of elements that are in the
         * paint list this frame, which does not include those elements removed
         * in this frame. So we loop the `prevList` to get the removed elements.
         */
        for (let i = this.__prevStartIndex; i < this.__prevEndIndex; ++i) {
            const el = prevList[i];
            /**
             * Consider the elements whose ancestors are invisible, they should
             * not be painted and their previous painting rects should be
             * cleared if they are rendered on the canvas (`__isRendered` being
             * true). `!shouldPaint` means the element is not brushed in this
             * frame.
             *
             * `!el.__zr` means it's removed from the storage.
             *
             * In conclusion, an element needs to repaint the previous painting
             * rect if and only if it's not painted this frame and was
             * previously painted on the canvas.
             */
            const shouldPaint = el && el.shouldBePainted(viewWidth, viewHeight, true, true);
            if (el && (!shouldPaint || !el.__zr) && el.__isRendered) {
                // el was removed
                const prevRect = el.getPrevPaintRect();
                if (prevRect) {
                    addRectToMergePool(prevRect);
                }
            }
        }

        // Merge intersected rects in the result
        let hasIntersections;
        do {
            hasIntersections = false;
            for (let i = 0; i < mergedRepaintRects.length;) {
                if (mergedRepaintRects[i].isZero()) {
                    mergedRepaintRects.splice(i, 1);
                    continue;
                }
                for (let j = i + 1; j < mergedRepaintRects.length;) {
                    if (mergedRepaintRects[i].intersect(mergedRepaintRects[j])) {
                        hasIntersections = true;
                        mergedRepaintRects[i].union(mergedRepaintRects[j]);
                        mergedRepaintRects.splice(j, 1);
                    }
                    else {
                        j++;
                    }
                }
                i++;
            }
        } while (hasIntersections);

        this._paintRects = mergedRepaintRects;

        return mergedRepaintRects;
    }

    /**
     * Get paint rects for debug usage.
     */
    debugGetPaintRects() {
        return (this._paintRects || []).slice();
    }

    resize(width: number, height: number) {
        const dpr = this.dpr;

        const dom = this.dom;
        const domStyle = dom.style;
        const domBack = this.domBack;

        if (domStyle) {
            domStyle.width = width + 'px';
            domStyle.height = height + 'px';
        }

        dom.width = width * dpr;
        dom.height = height * dpr;

        if (domBack) {
            domBack.width = width * dpr;
            domBack.height = height * dpr;

            if (dpr !== 1) {
                this.ctxBack.scale(dpr, dpr);
            }
        }
    }

    /**
     * 清空该层画布
     */
    clear(
        clearAll?: boolean,
        clearColor?: string | GradientObject | ImagePatternObject,
        repaintRects?: BoundingRect[]
    ) {
        const dom = this.dom;
        const ctx = this.ctx;
        const width = dom.width;
        const height = dom.height;

        clearColor = clearColor || this.clearColor;
        const haveMotionBLur = this.motionBlur && !clearAll;
        const lastFrameAlpha = this.lastFrameAlpha;

        const dpr = this.dpr;
        const self = this;

        if (haveMotionBLur) {
            if (!this.domBack) {
                this.createBackBuffer();
            }

            this.ctxBack.globalCompositeOperation = 'copy';
            this.ctxBack.drawImage(
                dom, 0, 0,
                width / dpr,
                height / dpr
            );
        }

        const domBack = this.domBack;

        function doClear(x: number, y: number, width: number, height: number) {
            ctx.clearRect(x, y, width, height);
            if (clearColor && clearColor !== 'transparent') {
                let clearColorGradientOrPattern;
                // Gradient
                if (util.isGradientObject(clearColor)) {
                    // shouldn't cache when clearColor is not global and size changed
                    const shouldCache = clearColor.global || (
                        (clearColor as InnerGradientObject).__width === width
                        && (clearColor as InnerGradientObject).__height === height
                    );
                    // Cache canvas gradient
                    clearColorGradientOrPattern = shouldCache
                        && (clearColor as InnerGradientObject).__canvasGradient
                        || getCanvasGradient(ctx, clearColor, {
                            x: 0,
                            y: 0,
                            width: width,
                            height: height
                        });

                    (clearColor as InnerGradientObject).__canvasGradient = clearColorGradientOrPattern;
                    (clearColor as InnerGradientObject).__width = width;
                    (clearColor as InnerGradientObject).__height = height;
                }
                // Pattern
                else if (util.isImagePatternObject(clearColor)) {
                    // scale pattern by dpr
                    clearColor.scaleX = clearColor.scaleX || dpr;
                    clearColor.scaleY = clearColor.scaleY || dpr;
                    clearColorGradientOrPattern = createCanvasPattern(
                        ctx, clearColor, {
                            dirty() {
                                self.setUnpainted();
                                self.painter.refresh();
                            }
                        }
                    );
                }
                ctx.save();
                ctx.fillStyle = clearColorGradientOrPattern || (clearColor as string);
                ctx.fillRect(x, y, width, height);
                ctx.restore();
            }

            if (haveMotionBLur) {
                ctx.save();
                ctx.globalAlpha = lastFrameAlpha;
                ctx.drawImage(domBack, x, y, width, height);
                ctx.restore();
            }
        };

        if (!repaintRects || haveMotionBLur) {
            // Clear the full canvas
            doClear(0, 0, width, height);
        }
        else if (repaintRects.length) {
            // Clear the repaint areas
            util.each(repaintRects, rect => {
                doClear(
                    rect.x * dpr,
                    rect.y * dpr,
                    rect.width * dpr,
                    rect.height * dpr
                );
            });
        }
    }

    // Interface of refresh
    refresh: (clearColor?: string | GradientObject | ImagePatternObject) => void

    // Interface of renderToCanvas in getRenderedCanvas
    renderToCanvas: (ctx: CanvasRenderingContext2D) => void

    // Events
    onclick: ElementEventCallback<unknown, this>
    ondblclick: ElementEventCallback<unknown, this>
    onmouseover: ElementEventCallback<unknown, this>
    onmouseout: ElementEventCallback<unknown, this>
    onmousemove: ElementEventCallback<unknown, this>
    onmousewheel: ElementEventCallback<unknown, this>
    onmousedown: ElementEventCallback<unknown, this>
    onmouseup: ElementEventCallback<unknown, this>
    oncontextmenu: ElementEventCallback<unknown, this>

    ondrag: ElementEventCallback<unknown, this>
    ondragstart: ElementEventCallback<unknown, this>
    ondragend: ElementEventCallback<unknown, this>
    ondragenter: ElementEventCallback<unknown, this>
    ondragleave: ElementEventCallback<unknown, this>
    ondragover: ElementEventCallback<unknown, this>
    ondrop: ElementEventCallback<unknown, this>
}
