/**
 * Base class of all displayable graphic objects
 */

import Element, {ElementProps, ElementStatePropNames, ElementAnimateConfig, ElementCommonState} from '../Element';
import BoundingRect from '../core/BoundingRect';
import { PropType, Dictionary, MapToType } from '../core/types';
import Path from './Path';
import { keys, extend, createObject } from '../core/util';
import Animator from '../animation/Animator';
import { REDRAW_BIT, STYLE_CHANGED_BIT } from './constants';

// type CalculateTextPositionResult = ReturnType<typeof calculateTextPosition>

const STYLE_MAGIC_KEY = '__zr_style_' + Math.round((Math.random() * 10));

export interface CommonStyleProps {
    shadowBlur?: number
    shadowOffsetX?: number
    shadowOffsetY?: number
    shadowColor?: string

    opacity?: number
    /**
     * https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/globalCompositeOperation
     */
    blend?: string
}

export const DEFAULT_COMMON_STYLE: CommonStyleProps = {
    shadowBlur: 0,
    shadowOffsetX: 0,
    shadowOffsetY: 0,
    shadowColor: '#000',
    opacity: 1,
    blend: 'source-over'
};

export const DEFAULT_COMMON_ANIMATION_PROPS: MapToType<DisplayableProps, boolean> = {
    style: {
        shadowBlur: true,
        shadowOffsetX: true,
        shadowOffsetY: true,
        shadowColor: true,
        opacity: true
    }
 };

(DEFAULT_COMMON_STYLE as any)[STYLE_MAGIC_KEY] = true;

export interface DisplayableProps extends ElementProps {
    style?: Dictionary<any>

    zlevel?: number
    z?: number
    z2?: number

    culling?: boolean

    // TODO list all cursors
    cursor?: string

    rectHover?: boolean

    progressive?: boolean

    incremental?: boolean

    ignoreCoarsePointer?: boolean

    batch?: boolean
    invisible?: boolean
}

type DisplayableKey = keyof DisplayableProps
type DisplayablePropertyType = PropType<DisplayableProps, DisplayableKey>

export type DisplayableStatePropNames = ElementStatePropNames | 'style' | 'z' | 'z2' | 'invisible';
export type DisplayableState = Pick<DisplayableProps, DisplayableStatePropNames> & ElementCommonState;

const PRIMARY_STATES_KEYS = ['z', 'z2', 'invisible'] as const;
const PRIMARY_STATES_KEYS_IN_HOVER_LAYER = ['invisible'] as const;

// eslint-disable-next-line @typescript-eslint/no-unused-vars
interface Displayable<Props extends DisplayableProps = DisplayableProps> {
    animate(key?: '', loop?: boolean): Animator<this>
    animate(key: 'style', loop?: boolean): Animator<this['style']>

    getState(stateName: string): DisplayableState
    ensureState(stateName: string): DisplayableState

    states: Dictionary<DisplayableState>
    stateProxy: (stateName: string) => DisplayableState
}

class Displayable<Props extends DisplayableProps = DisplayableProps> extends Element<Props> {

    /**
     * Whether the displayable object is visible. when it is true, the displayable object
     * is not drawn, but the mouse event can still trigger the object.
     */
    invisible: boolean

    z: number

    z2: number

    /**
     * The z level determines the displayable object can be drawn in which layer canvas.
     */
    zlevel: number

    /**
     * If enable culling
     */
    culling: boolean

    /**
     * Mouse cursor when hovered
     */
    cursor: string

    /**
     * If hover area is bounding rect
     */
    rectHover: boolean
    /**
     * For increamental rendering
     */
    incremental: boolean

    /**
     * Never increase to target size
     */
    ignoreCoarsePointer?: boolean

    style: Dictionary<any>

    protected _normalState: DisplayableState

    protected _rect: BoundingRect
    protected _paintRect: BoundingRect
    protected _prevPaintRect: BoundingRect

    dirtyRectTolerance: number

    /************* Properties will be inejected in other modules. *******************/

    // @deprecated.
    useHoverLayer?: boolean

    __hoverStyle?: CommonStyleProps

    // TODO use WeakMap?

    // Shapes for cascade clipping.
    // Can only be `null`/`undefined` or an non-empty array, MUST NOT be an empty array.
    // because it is easy to only using null to check whether clipPaths changed.
    __clipPaths?: Path[]

    // FOR CANVAS PAINTER
    __canvasFillGradient: CanvasGradient
    __canvasStrokeGradient: CanvasGradient
    __canvasFillPattern: CanvasPattern
    __canvasStrokePattern: CanvasPattern

    // FOR SVG PAINTER
    __svgEl: SVGElement

    constructor(props?: Props) {
        super(props);
    }

    protected _init(props?: Props) {
        // Init default properties
        const keysArr = keys(props);
        for (let i = 0; i < keysArr.length; i++) {
            const key = keysArr[i];
            if (key === 'style') {
                this.useStyle(props[key] as Props['style']);
            }
            else {
                super.attrKV(key as any, props[key]);
            }
        }
        // Give a empty style
        if (!this.style) {
            this.useStyle({});
        }
    }

    // Hook provided to developers.
    beforeBrush() {}
    afterBrush() {}

    // Hook provided to inherited classes.
    // Executed between beforeBrush / afterBrush
    innerBeforeBrush() {}
    innerAfterBrush() {}

    shouldBePainted(
        viewWidth: number,
        viewHeight: number,
        considerClipPath: boolean,
        considerAncestors: boolean
    ) {
        const m = this.transform;
        if (
            this.ignore
            // Ignore invisible element
            || this.invisible
            // Ignore transparent element
            || this.style.opacity === 0
            // Ignore culled element
            || (this.culling
                && isDisplayableCulled(this, viewWidth, viewHeight)
            )
            // Ignore scale 0 element, in some environment like node-canvas
            // Draw a scale 0 element can cause all following draw wrong
            // And setTransform with scale 0 will cause set back transform failed.
            || (m && !m[0] && !m[3])
        ) {
            return false;
        }

        if (considerClipPath && this.__clipPaths) {
            for (let i = 0; i < this.__clipPaths.length; ++i) {
                if (this.__clipPaths[i].isZeroArea()) {
                    return false;
                }
            }
        }

        if (considerAncestors && this.parent) {
            let parent = this.parent;
            while (parent) {
                if (parent.ignore) {
                    return false;
                }
                parent = parent.parent;
            }
        }

        return true;
    }

    /**
     * If displayable element contain coord x, y
     */
    contain(x: number, y: number) {
        return this.rectContain(x, y);
    }

    traverse<Context>(
        cb: (this: Context, el: this) => void,
        context?: Context
    ) {
        cb.call(context, this);
    }

    /**
     * If bounding rect of element contain coord x, y
     */
    rectContain(x: number, y: number) {
        const coord = this.transformCoordToLocal(x, y);
        const rect = this.getBoundingRect();
        return rect.contain(coord[0], coord[1]);
    }

    getPaintRect(): BoundingRect {
        let rect = this._paintRect;
        if (!this._paintRect || this.__dirty) {
            const transform = this.transform;
            const elRect = this.getBoundingRect();

            const style = this.style;
            const shadowSize = style.shadowBlur || 0;
            const shadowOffsetX = style.shadowOffsetX || 0;
            const shadowOffsetY = style.shadowOffsetY || 0;

            rect = this._paintRect || (this._paintRect = new BoundingRect(0, 0, 0, 0));
            if (transform) {
                BoundingRect.applyTransform(rect, elRect, transform);
            }
            else {
                rect.copy(elRect);
            }

            if (shadowSize || shadowOffsetX || shadowOffsetY) {
                rect.width += shadowSize * 2 + Math.abs(shadowOffsetX);
                rect.height += shadowSize * 2 + Math.abs(shadowOffsetY);
                rect.x = Math.min(rect.x, rect.x + shadowOffsetX - shadowSize);
                rect.y = Math.min(rect.y, rect.y + shadowOffsetY - shadowSize);

            }

            // For the accuracy tolerance of text height or line joint point
            const tolerance = this.dirtyRectTolerance;
            if (!rect.isZero()) {
                rect.x = Math.floor(rect.x - tolerance);
                rect.y = Math.floor(rect.y - tolerance);
                rect.width = Math.ceil(rect.width + 1 + tolerance * 2);
                rect.height = Math.ceil(rect.height + 1 + tolerance * 2);
            }
        }
        return rect;
    }

    setPrevPaintRect(paintRect: BoundingRect) {
        if (paintRect) {
            this._prevPaintRect = this._prevPaintRect || new BoundingRect(0, 0, 0, 0);
            this._prevPaintRect.copy(paintRect);
        }
        else {
            this._prevPaintRect = null;
        }
    }

    getPrevPaintRect(): BoundingRect {
        return this._prevPaintRect;
    }

    /**
     * Alias for animate('style')
     * @param loop
     */
    animateStyle(loop: boolean) {
        return this.animate('style', loop);
    }

    // Override updateDuringAnimation
    updateDuringAnimation(targetKey: string) {
        if (targetKey === 'style') {
            this.dirtyStyle();
        }
        else {
            this.markRedraw();
        }
    }

    attrKV(key: DisplayableKey, value: DisplayablePropertyType) {
        if (key !== 'style') {
            super.attrKV(key as keyof DisplayableProps, value);
        }
        else {
            if (!this.style) {
                this.useStyle(value as Dictionary<any>);
            }
            else {
                this.setStyle(value as Dictionary<any>);
            }
        }
    }

    setStyle(obj: Props['style']): this
    setStyle<T extends keyof Props['style']>(obj: T, value: Props['style'][T]): this
    setStyle(keyOrObj: keyof Props['style'] | Props['style'], value?: unknown): this {
        if (typeof keyOrObj === 'string') {
            this.style[keyOrObj] = value;
        }
        else {
            extend(this.style, keyOrObj as Props['style']);
        }
        this.dirtyStyle();
        return this;
    }

    // getDefaultStyleValue<T extends keyof Props['style']>(key: T): Props['style'][T] {
    //     // Default value is on the prototype.
    //     return this.style.prototype[key];
    // }

    dirtyStyle(notRedraw?: boolean) {
        if (!notRedraw) {
            this.markRedraw();
        }
        this.__dirty |= STYLE_CHANGED_BIT;
        // Clear bounding rect.
        if (this._rect) {
            this._rect = null;
        }
    }

    dirty() {
        this.dirtyStyle();
    }

    /**
     * Is style changed. Used with dirtyStyle.
     */
    styleChanged() {
        return !!(this.__dirty & STYLE_CHANGED_BIT);
    }

    /**
     * Mark style updated. Only useful when style is used for caching. Like in the text.
     */
    styleUpdated() {
        this.__dirty &= ~STYLE_CHANGED_BIT;
    }

    /**
     * Create a style object with default values in it's prototype.
     */
    createStyle(obj?: Props['style']) {
        return createObject(DEFAULT_COMMON_STYLE, obj);
    }

    /**
     * Replace style property.
     * It will create a new style if given obj is not a valid style object.
     */
     // PENDING should not createStyle if it's an style object.
    useStyle(obj: Props['style']) {
        if (!obj[STYLE_MAGIC_KEY]) {
            obj = this.createStyle(obj);
        }
        if (this.__inHover) {
            this.__hoverStyle = obj;    // Not affect exists style.
        }
        else {
            this.style = obj;
        }
        this.dirtyStyle();
    }

    /**
     * Determine if an object is a valid style object.
     * Which means it is created by `createStyle.`
     *
     * A valid style object will have all default values in it's prototype.
     * To avoid get null/undefined values.
     */
    isStyleObject(obj: Props['style']) {
        return obj[STYLE_MAGIC_KEY];
    }

    protected _innerSaveToNormal(toState: DisplayableState) {
        super._innerSaveToNormal(toState);

        const normalState = this._normalState;
        if (toState.style && !normalState.style) {
            // Clone style object.
            // TODO: Only save changed style.
            normalState.style = this._mergeStyle(this.createStyle(), this.style);
        }

        this._savePrimaryToNormal(toState, normalState, PRIMARY_STATES_KEYS);
    }

    protected _applyStateObj(
        stateName: string,
        state: DisplayableState,
        normalState: DisplayableState,
        keepCurrentStates: boolean,
        transition: boolean,
        animationCfg: ElementAnimateConfig
    ) {
        super._applyStateObj(stateName, state, normalState, keepCurrentStates, transition, animationCfg);

        const needsRestoreToNormal = !(state && keepCurrentStates);
        let targetStyle: Props['style'];
        if (state && state.style) {
            // Only animate changed properties.
            if (transition) {
                if (keepCurrentStates) {
                    targetStyle = state.style;
                }
                else {
                    targetStyle = this._mergeStyle(this.createStyle(), normalState.style);
                    this._mergeStyle(targetStyle, state.style);
                }
            }
            else {
                targetStyle = this._mergeStyle(
                    this.createStyle(),
                    keepCurrentStates ? this.style : normalState.style
                );
                this._mergeStyle(targetStyle, state.style);
            }
        }
        else if (needsRestoreToNormal) {
            targetStyle = normalState.style;
        }

        if (targetStyle) {
            if (transition) {
                // Clone a new style. Not affect the original one.
                const sourceStyle = this.style;

                this.style = this.createStyle(needsRestoreToNormal ? {} : sourceStyle);
                // const sourceStyle = this.style = this.createStyle(this.style);

                if (needsRestoreToNormal) {
                    const changedKeys = keys(sourceStyle);
                    for (let i = 0; i < changedKeys.length; i++) {
                        const key = changedKeys[i];
                        if (key in targetStyle) {   // Not use `key == null` because == null may means no stroke/fill.
                            // Pick out from prototype. Or the property won't be animated.
                            (targetStyle as any)[key] = targetStyle[key];
                            // Omit the property has no default value.
                            (this.style as any)[key] = sourceStyle[key];
                        }
                    }
                }

                // If states is switched twice in ONE FRAME, for example:
                // one property(for example shadowBlur) changed from default value to a specifed value,
                // then switched back in immediately. this.style may don't set this property yet when switching back.
                // It won't treat it as an changed property when switching back. And it won't be animated.
                // So here we make sure the properties will be animated from default value to a specifed value are set.
                const targetKeys = keys(targetStyle);
                for (let i = 0; i < targetKeys.length; i++) {
                    const key = targetKeys[i];
                    this.style[key] = this.style[key];
                }

                this._transitionState(stateName, {
                    style: targetStyle
                } as Props, animationCfg, this.getAnimationStyleProps() as MapToType<Props, boolean>);
            }
            else {
                this.useStyle(targetStyle);
            }
        }

        // Don't change z, z2 for element moved into hover layer.
        // It's not necessary and will cause paint list order changed.
        const statesKeys = this.__inHover ? PRIMARY_STATES_KEYS_IN_HOVER_LAYER : PRIMARY_STATES_KEYS;
        for (let i = 0; i < statesKeys.length; i++) {
            let key = statesKeys[i];
            if (state && state[key] != null) {
                // Replace if it exist in target state
                (this as any)[key] = state[key];
            }
            else if (needsRestoreToNormal) {
                // Restore to normal state
                if (normalState[key] != null) {
                    (this as any)[key] = normalState[key];
                }
            }
        }
    }

    protected _mergeStates(states: DisplayableState[]) {
        const mergedState = super._mergeStates(states) as DisplayableState;
        let mergedStyle: Props['style'];
        for (let i = 0; i < states.length; i++) {
            const state = states[i];
            if (state.style) {
                mergedStyle = mergedStyle || {};
                this._mergeStyle(mergedStyle, state.style);
            }
        }
        if (mergedStyle) {
            mergedState.style = mergedStyle;
        }
        return mergedState;
    }

    protected _mergeStyle(
        targetStyle: CommonStyleProps,
        sourceStyle: CommonStyleProps
    ) {
        extend(targetStyle, sourceStyle);
        return targetStyle;
    }

    getAnimationStyleProps() {
        return DEFAULT_COMMON_ANIMATION_PROPS;
    }

    /**
     * The string value of `textPosition` needs to be calculated to a real postion.
     * For example, `'inside'` is calculated to `[rect.width/2, rect.height/2]`
     * by default. See `contain/text.js#calculateTextPosition` for more details.
     * But some coutom shapes like "pin", "flag" have center that is not exactly
     * `[width/2, height/2]`. So we provide this hook to customize the calculation
     * for those shapes. It will be called if the `style.textPosition` is a string.
     * @param out Prepared out object. If not provided, this method should
     *        be responsible for creating one.
     * @param style
     * @param rect {x, y, width, height}
     * @return out The same as the input out.
     *         {
     *             x: number. mandatory.
     *             y: number. mandatory.
     *             textAlign: string. optional. use style.textAlign by default.
     *             textVerticalAlign: string. optional. use style.textVerticalAlign by default.
     *         }
     */
    // calculateTextPosition: (out: CalculateTextPositionResult, style: Dictionary<any>, rect: RectLike) => CalculateTextPositionResult

    protected static initDefaultProps = (function () {
        const dispProto = Displayable.prototype;
        dispProto.type = 'displayable';
        dispProto.invisible = false;
        dispProto.z = 0;
        dispProto.z2 = 0;
        dispProto.zlevel = 0;
        dispProto.culling = false;
        dispProto.cursor = 'pointer';
        dispProto.rectHover = false;
        dispProto.incremental = false;
        dispProto._rect = null;
        dispProto.dirtyRectTolerance = 0;

        dispProto.__dirty = REDRAW_BIT | STYLE_CHANGED_BIT;
    })()
}

const tmpRect = new BoundingRect(0, 0, 0, 0);
const viewRect = new BoundingRect(0, 0, 0, 0);
function isDisplayableCulled(el: Displayable, width: number, height: number) {
    tmpRect.copy(el.getBoundingRect());
    if (el.transform) {
        tmpRect.applyTransform(el.transform);
    }
    viewRect.width = width;
    viewRect.height = height;
    return !tmpRect.intersect(viewRect);
}

export default Displayable;