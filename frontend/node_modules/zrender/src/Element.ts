import Transformable, {TRANSFORMABLE_PROPS, TransformProp} from './core/Transformable';
import { AnimationEasing } from './animation/easing';
import Animator, {cloneValue} from './animation/Animator';
import { ZRenderType } from './zrender';
import {
    Dictionary, ElementEventName, ZRRawEvent, BuiltinTextPosition, AllPropTypes,
    TextVerticalAlign, TextAlign, MapToType
} from './core/types';
import Path from './graphic/Path';
import BoundingRect, { RectLike } from './core/BoundingRect';
import Eventful from './core/Eventful';
import ZRText, { DefaultTextStyle } from './graphic/Text';
import { calculateTextPosition, TextPositionCalculationResult, parsePercent } from './contain/text';
import {
    guid,
    isObject,
    keys,
    extend,
    indexOf,
    logError,
    mixin,
    isArrayLike,
    isTypedArray,
    isGradientObject,
    filter,
    reduce
} from './core/util';
import Polyline from './graphic/shape/Polyline';
import Group from './graphic/Group';
import Point from './core/Point';
import { LIGHT_LABEL_COLOR, DARK_LABEL_COLOR } from './config';
import { parse, stringify } from './tool/color';
import { REDRAW_BIT } from './graphic/constants';

export interface ElementAnimateConfig {
    duration?: number
    delay?: number
    easing?: AnimationEasing
    during?: (percent: number) => void

    // `done` will be called when all of the animations of the target props are
    // "done" or "aborted", and at least one "done" happened.
    // Common cases: animations declared, but some of them are aborted (e.g., by state change).
    // The calling of `animationTo` done rather than aborted if at least one done happened.
    done?: Function
    // `aborted` will be called when all of the animations of the target props are "aborted".
    aborted?: Function

    scope?: string
    /**
     * If force animate
     * Prevent stop animation and callback
     * immediently when target values are the same as current values.
     */
    force?: boolean
    /**
     * If use additive animation.
     */
    additive?: boolean
    /**
     * If set to final state before animation started.
     * It can be useful if something you want to calcuate depends on the final state of element.
     * Like bounding rect for text layouting.
     *
     * Only available in animateTo
     */
    setToFinal?: boolean
}

export interface ElementTextConfig {
    /**
     * Position relative to the element bounding rect
     * @default 'inside'
     */
    position?: BuiltinTextPosition | (number | string)[]

    /**
     * Rotation of the label.
     */
    rotation?: number

    /**
     * Rect that text will be positioned.
     * Default to be the rect of element.
     */
    layoutRect?: RectLike

    /**
     * Offset of the label.
     * The difference of offset and position is that it will be applied
     * in the rotation
     */
    offset?: number[]

    /**
     * Origin or rotation. Which is relative to the bounding box of the attached element.
     * Can be percent value. Relative to the bounding box.
     * If specified center. It will be center of the bounding box.
     *
     * Only available when position and rotation are both set.
     */
    origin?: (number | string)[] | 'center'

    /**
     * Distance to the rect
     * @default 5
     */
    distance?: number

    /**
     * If use local user space. Which will apply host's transform
     * @default false
     */
    local?: boolean

    /**
     * `insideFill` is a color string or left empty.
     * If a `textContent` is "inside", its final `fill` will be picked by this priority:
     * `textContent.style.fill` > `textConfig.insideFill` > "auto-calculated-fill"
     * In most cases, "auto-calculated-fill" is white.
     */
    insideFill?: string

    /**
     * `insideStroke` is a color string or left empty.
     * If a `textContent` is "inside", its final `stroke` will be picked by this priority:
     * `textContent.style.stroke` > `textConfig.insideStroke` > "auto-calculated-stroke"
     *
     * The rule of getting "auto-calculated-stroke":
     * If (A) the `fill` is specified in style (either in `textContent.style` or `textContent.style.rich`)
     * or (B) needed to draw text background (either defined in `textContent.style` or `textContent.style.rich`)
     * "auto-calculated-stroke" will be null.
     * Otherwise, "auto-calculated-stroke" will be the same as `fill` of this element if possible, or null.
     *
     * The reason of (A) is not decisive:
     * 1. If users specify `fill` in style and still use "auto-calculated-stroke", the effect
     * is not good and unexpected in some cases. It not easy and seams uncessary to auto calculate
     * a proper `stroke` for the given `fill`, since they can specify `stroke` themselve.
     * 2. Backward compat.
     */
    insideStroke?: string

    /**
     * `outsideFill` is a color string or left empty.
     * If a `textContent` is "inside", its final `fill` will be picked by this priority:
     * `textContent.style.fill` > `textConfig.outsideFill` > #000
     */
    outsideFill?: string

    /**
     * `outsideStroke` is a color string or left empth.
     * If a `textContent` is not "inside", its final `stroke` will be picked by this priority:
     * `textContent.style.stroke` > `textConfig.outsideStroke` > "auto-calculated-stroke"
     *
     * The rule of getting "auto-calculated-stroke":
     * If (A) the `fill` is specified in style (either in `textContent.style` or `textContent.style.rich`)
     * or (B) needed to draw text background (either defined in `textContent.style` or `textContent.style.rich`)
     * "auto-calculated-stroke" will be null.
     * Otherwise, "auto-calculated-stroke" will be a neer white color to distinguish "front end"
     * label with messy background (like other text label, line or other graphic).
     */
    outsideStroke?: string

    /**
     * Tell zrender I can sure this text is inside or not.
     * In case position is not using builtin `inside` hints.
     */
    inside?: boolean
}
export interface ElementTextGuideLineConfig {
    /**
     * Anchor for text guide line.
     * Notice: Won't work
     */
    anchor?: Point

    /**
     * If above the target element.
     */
    showAbove?: boolean

    /**
     * Candidates of connectors. Used when autoCalculate is true and anchor is not specified.
     */
    candidates?: ('left' | 'top' | 'right' | 'bottom')[]
}

export interface ElementEvent {
    type: ElementEventName,
    event: ZRRawEvent,
    // target can only be an element that is not silent.
    target: Element,
    // topTarget can be a silent element.
    topTarget: Element,
    cancelBubble: boolean,
    offsetX: number,
    offsetY: number,
    gestureEvent: string,
    pinchX: number,
    pinchY: number,
    pinchScale: number,
    wheelDelta: number,
    zrByTouch: boolean,
    which: number,
    stop: (this: ElementEvent) => void
}

export type ElementEventCallback<Ctx, Impl> = (
    this: CbThis<Ctx, Impl>, e: ElementEvent
) => boolean | void
type CbThis<Ctx, Impl> = unknown extends Ctx ? Impl : Ctx;

interface ElementEventHandlerProps {
    // Events
    onclick: ElementEventCallback<unknown, unknown>
    ondblclick: ElementEventCallback<unknown, unknown>
    onmouseover: ElementEventCallback<unknown, unknown>
    onmouseout: ElementEventCallback<unknown, unknown>
    onmousemove: ElementEventCallback<unknown, unknown>
    onmousewheel: ElementEventCallback<unknown, unknown>
    onmousedown: ElementEventCallback<unknown, unknown>
    onmouseup: ElementEventCallback<unknown, unknown>
    oncontextmenu: ElementEventCallback<unknown, unknown>

    ondrag: ElementEventCallback<unknown, unknown>
    ondragstart: ElementEventCallback<unknown, unknown>
    ondragend: ElementEventCallback<unknown, unknown>
    ondragenter: ElementEventCallback<unknown, unknown>
    ondragleave: ElementEventCallback<unknown, unknown>
    ondragover: ElementEventCallback<unknown, unknown>
    ondrop: ElementEventCallback<unknown, unknown>
}

export interface ElementProps extends Partial<ElementEventHandlerProps>, Partial<Pick<Transformable, TransformProp>> {
    name?: string
    ignore?: boolean
    isGroup?: boolean
    draggable?: boolean | 'horizontal' | 'vertical'

    silent?: boolean

    ignoreClip?: boolean
    globalScaleRatio?: number

    textConfig?: ElementTextConfig
    textContent?: ZRText

    clipPath?: Path
    drift?: Element['drift']

    extra?: Dictionary<unknown>

    // For echarts animation.
    anid?: string
}

// Properties can be used in state.
export const PRESERVED_NORMAL_STATE = '__zr_normal__';
// export const PRESERVED_MERGED_STATE = '__zr_merged__';

const PRIMARY_STATES_KEYS = (TRANSFORMABLE_PROPS as any).concat(['ignore']) as [TransformProp, 'ignore'];
const DEFAULT_ANIMATABLE_MAP = reduce(TRANSFORMABLE_PROPS, (obj, key) => {
    obj[key] = true;
    return obj;
}, {ignore: false} as Partial<Record<ElementStatePropNames, boolean>>);

export type ElementStatePropNames = (typeof PRIMARY_STATES_KEYS)[number] | 'textConfig';
export type ElementState = Pick<ElementProps, ElementStatePropNames> & ElementCommonState

export type ElementCommonState = {
    hoverLayer?: boolean
}

export type ElementCalculateTextPosition = (
    out: TextPositionCalculationResult,
    style: ElementTextConfig,
    rect: RectLike
) => TextPositionCalculationResult;

let tmpTextPosCalcRes = {} as TextPositionCalculationResult;
let tmpBoundingRect = new BoundingRect(0, 0, 0, 0);

// eslint-disable-next-line @typescript-eslint/no-unused-vars
interface Element<Props extends ElementProps = ElementProps> extends Transformable,
    Eventful<{
        [key in ElementEventName]: (e: ElementEvent) => void | boolean
    } & {
        [key in string]: (...args: any) => void | boolean
    }>,
    ElementEventHandlerProps {
}

class Element<Props extends ElementProps = ElementProps> {

    id: number = guid()
    /**
     * Element type
     */
    type: string

    /**
     * Element name
     */
    name: string

    /**
     * If ignore drawing and events of the element object
     */
    ignore: boolean

    /**
     * Whether to respond to mouse events.
     */
    silent: boolean

    /**
     * 是否是 Group
     */
    isGroup: boolean

    /**
     * Whether it can be dragged.
     */
    draggable: boolean | 'horizontal' | 'vertical'

    /**
     * Whether is it dragging.
     */
    dragging: boolean

    parent: Group

    animators: Animator<any>[] = []

    /**
     * If ignore clip from it's parent or hosts.
     * Applied on itself and all it's children.
     *
     * NOTE: It won't affect the clipPath set on the children.
     */
    ignoreClip: boolean

    /**
     * If element is used as a component of other element.
     */
    __hostTarget: Element

    /**
     * ZRender instance will be assigned when element is associated with zrender
     */
    __zr: ZRenderType

    /**
     * Dirty bits.
     * From which painter will determine if this displayable object needs brush.
     */
    __dirty: number

    /**
     * If element was painted on the screen
     */
    __isRendered: boolean;

    /**
     * If element has been moved to the hover layer.
     *
     * If so, dirty will only trigger the zrender refresh hover layer
     */
    __inHover: boolean

    /**
     * path to clip the elements and its children, if it is a group.
     * @see http://www.w3.org/TR/2dcontext/#clipping-region
     */
    private _clipPath?: Path

    /**
     * Attached text element.
     * `position`, `style.textAlign`, `style.textVerticalAlign`
     * of element will be ignored if textContent.position is set
     */
    private _textContent?: ZRText

    /**
     * Text guide line.
     */
    private _textGuide?: Polyline

    /**
     * Config of textContent. Inlcuding layout, color, ...etc.
     */
    textConfig?: ElementTextConfig

    /**
     * Config for guide line calculating.
     *
     * NOTE: This is just a property signature. READ and WRITE are all done in echarts.
     */
    textGuideLineConfig?: ElementTextGuideLineConfig

    // FOR ECHARTS
    /**
     * Id for mapping animation
     */
    anid: string

    extra: Dictionary<unknown>

    currentStates?: string[] = []
    // prevStates is for storager in echarts.
    prevStates?: string[]
    /**
     * Store of element state.
     * '__normal__' key is preserved for default properties.
     */
    states: Dictionary<ElementState> = {}

    /**
     * Animation config applied on state switching.
     */
    stateTransition: ElementAnimateConfig

    /**
     * Proxy function for getting state with given stateName.
     * ZRender will first try to get with stateProxy. Then find from states if stateProxy returns nothing
     *
     * targetStates will be given in useStates
     */
    stateProxy?: (stateName: string, targetStates?: string[]) => ElementState

    protected _normalState: ElementState

    // Temporary storage for inside text color configuration.
    private _innerTextDefaultStyle: DefaultTextStyle

    constructor(props?: Props) {
        this._init(props);
    }

    protected _init(props?: Props) {
        // Init default properties
        this.attr(props);
    }

    /**
     * Drift element
     * @param {number} dx dx on the global space
     * @param {number} dy dy on the global space
     */
    drift(dx: number, dy: number, e?: ElementEvent) {
        switch (this.draggable) {
            case 'horizontal':
                dy = 0;
                break;
            case 'vertical':
                dx = 0;
                break;
        }

        let m = this.transform;
        if (!m) {
            m = this.transform = [1, 0, 0, 1, 0, 0];
        }
        m[4] += dx;
        m[5] += dy;

        this.decomposeTransform();
        this.markRedraw();
    }

    /**
     * Hook before update
     */
    beforeUpdate() {}
    /**
     * Hook after update
     */
    afterUpdate() {}
    /**
     * Update each frame
     */
    update() {
        this.updateTransform();

        if (this.__dirty) {
            this.updateInnerText();
        }
    }

    updateInnerText(forceUpdate?: boolean) {
        // Update textContent
        const textEl = this._textContent;
        if (textEl && (!textEl.ignore || forceUpdate)) {
            if (!this.textConfig) {
                this.textConfig = {};
            }
            const textConfig = this.textConfig;
            const isLocal = textConfig.local;
            const innerTransformable = textEl.innerTransformable;

            let textAlign: TextAlign;
            let textVerticalAlign: TextVerticalAlign;

            let textStyleChanged = false;

            // Apply host's transform.
            innerTransformable.parent = isLocal ? this as unknown as Group : null;

            let innerOrigin = false;

            // Reset x/y/rotation
            innerTransformable.copyTransform(textEl);

            // Force set attached text's position if `position` is in config.
            if (textConfig.position != null) {
                let layoutRect = tmpBoundingRect;
                if (textConfig.layoutRect) {
                    layoutRect.copy(textConfig.layoutRect);
                }
                else {
                    layoutRect.copy(this.getBoundingRect());
                }
                if (!isLocal) {
                    layoutRect.applyTransform(this.transform);
                }

                if (this.calculateTextPosition) {
                    this.calculateTextPosition(tmpTextPosCalcRes, textConfig, layoutRect);
                }
                else {
                    calculateTextPosition(tmpTextPosCalcRes, textConfig, layoutRect);
                }

                // TODO Should modify back if textConfig.position is set to null again.
                // Or textContent is detached.
                innerTransformable.x = tmpTextPosCalcRes.x;
                innerTransformable.y = tmpTextPosCalcRes.y;

                // User specified align/verticalAlign has higher priority, which is
                // useful in the case that attached text is rotated 90 degree.
                textAlign = tmpTextPosCalcRes.align;
                textVerticalAlign = tmpTextPosCalcRes.verticalAlign;

                const textOrigin = textConfig.origin;
                if (textOrigin && textConfig.rotation != null) {
                    let relOriginX;
                    let relOriginY;
                    if (textOrigin === 'center') {
                        relOriginX = layoutRect.width * 0.5;
                        relOriginY = layoutRect.height * 0.5;
                    }
                    else {
                        relOriginX = parsePercent(textOrigin[0], layoutRect.width);
                        relOriginY = parsePercent(textOrigin[1], layoutRect.height);
                    }

                    innerOrigin = true;
                    innerTransformable.originX = -innerTransformable.x + relOriginX + (isLocal ? 0 : layoutRect.x);
                    innerTransformable.originY = -innerTransformable.y + relOriginY + (isLocal ? 0 : layoutRect.y);
                }
            }


            if (textConfig.rotation != null) {
                innerTransformable.rotation = textConfig.rotation;
            }

            // TODO
            const textOffset = textConfig.offset;
            if (textOffset) {
                innerTransformable.x += textOffset[0];
                innerTransformable.y += textOffset[1];

                // Not change the user set origin.
                if (!innerOrigin) {
                    innerTransformable.originX = -textOffset[0];
                    innerTransformable.originY = -textOffset[1];
                }
            }

            // Calculate text color
            const isInside = textConfig.inside == null  // Force to be inside or not.
                ? (typeof textConfig.position === 'string' && textConfig.position.indexOf('inside') >= 0)
                : textConfig.inside;
            const innerTextDefaultStyle = this._innerTextDefaultStyle || (this._innerTextDefaultStyle = {});

            let textFill;
            let textStroke;
            let autoStroke;
            if (isInside && this.canBeInsideText()) {
                // In most cases `textContent` need this "auto" strategy.
                // So by default be 'auto'. Otherwise users need to literally
                // set `insideFill: 'auto', insideStroke: 'auto'` each time.
                textFill = textConfig.insideFill;
                textStroke = textConfig.insideStroke;

                if (textFill == null || textFill === 'auto') {
                    textFill = this.getInsideTextFill();
                }
                if (textStroke == null || textStroke === 'auto') {
                    textStroke = this.getInsideTextStroke(textFill);
                    autoStroke = true;
                }
            }
            else {
                textFill = textConfig.outsideFill;
                textStroke = textConfig.outsideStroke;

                if (textFill == null || textFill === 'auto') {
                    textFill = this.getOutsideFill();
                }
                // By default give a stroke to distinguish "front end" label with
                // messy background (like other text label, line or other graphic).
                // If textContent.style.fill specified, this auto stroke will not be used.
                if (textStroke == null || textStroke === 'auto') {
                    // If some time need to customize the default stroke getter,
                    // add some kind of override method.
                    textStroke = this.getOutsideStroke(textFill);
                    autoStroke = true;
                }
            }
            // Default `textFill` should must have a value to ensure text can be displayed.
            textFill = textFill || '#000';

            if (textFill !== innerTextDefaultStyle.fill
                || textStroke !== innerTextDefaultStyle.stroke
                || autoStroke !== innerTextDefaultStyle.autoStroke
                || textAlign !== innerTextDefaultStyle.align
                || textVerticalAlign !== innerTextDefaultStyle.verticalAlign
            ) {

                textStyleChanged = true;

                innerTextDefaultStyle.fill = textFill;
                innerTextDefaultStyle.stroke = textStroke;
                innerTextDefaultStyle.autoStroke = autoStroke;
                innerTextDefaultStyle.align = textAlign;
                innerTextDefaultStyle.verticalAlign = textVerticalAlign;

                textEl.setDefaultTextStyle(innerTextDefaultStyle);
            }

            // Mark textEl to update transform.
            // DON'T use markRedraw. It will cause Element itself to dirty again.
            textEl.__dirty |= REDRAW_BIT;

            if (textStyleChanged) {
                // Only mark style dirty if necessary. Update ZRText is costly.
                textEl.dirtyStyle(true);
            }
        }
    }

    protected canBeInsideText() {
        return true;
    }

    protected getInsideTextFill(): string | undefined {
        return '#fff';
    }

    protected getInsideTextStroke(textFill: string): string | undefined {
        return '#000';
    }

    protected getOutsideFill(): string | undefined {
        return this.__zr && this.__zr.isDarkMode() ? LIGHT_LABEL_COLOR : DARK_LABEL_COLOR;
    }

    protected getOutsideStroke(textFill: string): string {
        const backgroundColor = this.__zr && this.__zr.getBackgroundColor();
        let colorArr = typeof backgroundColor === 'string' && parse(backgroundColor as string);
        if (!colorArr) {
            colorArr = [255, 255, 255, 1];
        }
        // Assume blending on a white / black(dark) background.
        const alpha = colorArr[3];
        const isDark = this.__zr.isDarkMode();
        for (let i = 0; i < 3; i++) {
            colorArr[i] = colorArr[i] * alpha + (isDark ? 0 : 255) * (1 - alpha);
        }
        colorArr[3] = 1;
        return stringify(colorArr, 'rgba');
    }

    traverse<Context>(
        cb: (this: Context, el: Element<Props>) => void,
        context?: Context
    ) {}

    protected attrKV(key: string, value: unknown) {
        if (key === 'textConfig') {
            this.setTextConfig(value as ElementTextConfig);
        }
        else if (key === 'textContent') {
            this.setTextContent(value as ZRText);
        }
        else if (key === 'clipPath') {
            this.setClipPath(value as Path);
        }
        else if (key === 'extra') {
            this.extra = this.extra || {};
            extend(this.extra, value);
        }
        else {
            (this as any)[key] = value;
        }
    }

    /**
     * Hide the element
     */
    hide() {
        this.ignore = true;
        this.markRedraw();
    }

    /**
     * Show the element
     */
    show() {
        this.ignore = false;
        this.markRedraw();
    }

    attr(keyOrObj: Props): this
    attr<T extends keyof Props>(keyOrObj: T, value: Props[T]): this
    attr(keyOrObj: keyof Props | Props, value?: unknown): this {
        if (typeof keyOrObj === 'string') {
            this.attrKV(keyOrObj as keyof ElementProps, value as AllPropTypes<ElementProps>);
        }
        else if (isObject(keyOrObj)) {
            let obj = keyOrObj as object;
            let keysArr = keys(obj);
            for (let i = 0; i < keysArr.length; i++) {
                let key = keysArr[i];
                this.attrKV(key as keyof ElementProps, keyOrObj[key]);
            }
        }
        this.markRedraw();
        return this;
    }

    // Save current state to normal
    saveCurrentToNormalState(toState: ElementState) {
        this._innerSaveToNormal(toState);

        // If we are switching from normal to other state during animation.
        // We need to save final value of animation to the normal state. Not interpolated value.
        const normalState = this._normalState;
        for (let i = 0; i < this.animators.length; i++) {
            const animator = this.animators[i];
            const fromStateTransition = animator.__fromStateTransition;
            // Ignore animation from state transition(except normal).
            // Ignore loop animation.
            if (animator.getLoop() || fromStateTransition && fromStateTransition !== PRESERVED_NORMAL_STATE) {
                continue;
            }

            const targetName = animator.targetName;
            // Respecting the order of animation if multiple animator is
            // animating on the same property(If additive animation is used)
            const target = targetName
                ? (normalState as any)[targetName] : normalState;
            // Only save keys that are changed by the states.
            animator.saveTo(target);
        }
    }

    protected _innerSaveToNormal(toState: ElementState) {
        let normalState = this._normalState;
        if (!normalState) {
            // Clear previous stored normal states when switching from normalState to otherState.
            normalState = this._normalState = {};
        }
        if (toState.textConfig && !normalState.textConfig) {
            normalState.textConfig = this.textConfig;
        }

        this._savePrimaryToNormal(toState, normalState, PRIMARY_STATES_KEYS);
    }

    protected _savePrimaryToNormal(
        toState: Dictionary<any>, normalState: Dictionary<any>, primaryKeys: readonly string[]
    ) {
        for (let i = 0; i < primaryKeys.length; i++) {
            let key = primaryKeys[i];
            // Only save property that will be changed by toState
            // and has not been saved to normalState yet.
            if (toState[key] != null && !(key in normalState)) {
                (normalState as any)[key] = (this as any)[key];
            }
        }
    }

    /**
     * If has any state.
     */
    hasState() {
        return this.currentStates.length > 0;
    }

    /**
     * Get state object
     */
    getState(name: string) {
        return this.states[name];
    }


    /**
     * Ensure state exists. If not, will create one and return.
     */
    ensureState(name: string) {
        const states = this.states;
        if (!states[name]) {
            states[name] = {};
        }
        return states[name];
    }

    /**
     * Clear all states.
     */
    clearStates(noAnimation?: boolean) {
        this.useState(PRESERVED_NORMAL_STATE, false, noAnimation);
        // TODO set _normalState to null?
    }
    /**
     * Use state. State is a collection of properties.
     * Will return current state object if state exists and stateName has been changed.
     *
     * @param stateName State name to be switched to
     * @param keepCurrentState If keep current states.
     *      If not, it will inherit from the normal state.
     */
    useState(stateName: string, keepCurrentStates?: boolean, noAnimation?: boolean, forceUseHoverLayer?: boolean) {
        // Use preserved word __normal__
        // TODO: Only restore changed properties when restore to normal???
        const toNormalState = stateName === PRESERVED_NORMAL_STATE;
        const hasStates = this.hasState();

        if (!hasStates && toNormalState) {
            // If switched from normal to normal.
            return;
        }

        const currentStates = this.currentStates;
        const animationCfg = this.stateTransition;

        // No need to change in following cases:
        // 1. Keep current states. and already being applied before.
        // 2. Don't keep current states. And new state is same with the only one exists state.
        if (indexOf(currentStates, stateName) >= 0 && (keepCurrentStates || currentStates.length === 1)) {
            return;
        }

        let state;
        if (this.stateProxy && !toNormalState) {
            state = this.stateProxy(stateName);
        }

        if (!state) {
            state = (this.states && this.states[stateName]);
        }

        if (!state && !toNormalState) {
            logError(`State ${stateName} not exists.`);
            return;
        }

        if (!toNormalState) {
            this.saveCurrentToNormalState(state);
        }

        const useHoverLayer = !!((state && state.hoverLayer) || forceUseHoverLayer);

        if (useHoverLayer) {
            // Enter hover layer before states update.
            this._toggleHoverLayerFlag(true);
        }

        this._applyStateObj(
            stateName,
            state,
            this._normalState,
            keepCurrentStates,
            !noAnimation && !this.__inHover && animationCfg && animationCfg.duration > 0,
            animationCfg
        );

        // Also set text content.
        const textContent = this._textContent;
        const textGuide = this._textGuide;
        if (textContent) {
            // Force textContent use hover layer if self is using it.
            textContent.useState(stateName, keepCurrentStates, noAnimation, useHoverLayer);
        }
        if (textGuide) {
            textGuide.useState(stateName, keepCurrentStates, noAnimation, useHoverLayer);
        }

        if (toNormalState) {
            // Clear state
            this.currentStates = [];
            // Reset normal state.
            this._normalState = {};
        }
        else {
            if (!keepCurrentStates) {
                this.currentStates = [stateName];
            }
            else {
                this.currentStates.push(stateName);
            }
        }

        // Update animating target to the new object after state changed.
        this._updateAnimationTargets();

        this.markRedraw();

        if (!useHoverLayer && this.__inHover) {
            // Leave hover layer after states update and markRedraw.
            this._toggleHoverLayerFlag(false);
            // NOTE: avoid unexpected refresh when moving out from hover layer!!
            // Only clear from hover layer.
            this.__dirty &= ~REDRAW_BIT;
        }

        // Return used state.
        return state;
    }

    /**
     * Apply multiple states.
     * @param states States list.
     */
    useStates(states: string[], noAnimation?: boolean, forceUseHoverLayer?: boolean) {
        if (!states.length) {
            this.clearStates();
        }
        else {
            const stateObjects: ElementState[] = [];
            const currentStates = this.currentStates;
            const len = states.length;
            let notChange = len === currentStates.length;
            if (notChange) {
                for (let i = 0; i < len; i++) {
                    if (states[i] !== currentStates[i]) {
                        notChange = false;
                        break;
                    }
                }
            }
            if (notChange) {
                return;
            }

            for (let i = 0; i < len; i++) {
                const stateName = states[i];
                let stateObj: ElementState;
                if (this.stateProxy) {
                    stateObj = this.stateProxy(stateName, states);
                }
                if (!stateObj) {
                    stateObj = this.states[stateName];
                }
                if (stateObj) {
                    stateObjects.push(stateObj);
                }
            }

            const lastStateObj = stateObjects[len - 1];
            const useHoverLayer = !!((lastStateObj && lastStateObj.hoverLayer) || forceUseHoverLayer);
            if (useHoverLayer) {
                // Enter hover layer before states update.
                this._toggleHoverLayerFlag(true);
            }

            const mergedState = this._mergeStates(stateObjects);
            const animationCfg = this.stateTransition;

            this.saveCurrentToNormalState(mergedState);

            this._applyStateObj(
                states.join(','),
                mergedState,
                this._normalState,
                false,
                !noAnimation && !this.__inHover && animationCfg && animationCfg.duration > 0,
                animationCfg
            );

            const textContent = this._textContent;
            const textGuide = this._textGuide;
            if (textContent) {
                textContent.useStates(states, noAnimation, useHoverLayer);
            }
            if (textGuide) {
                textGuide.useStates(states, noAnimation, useHoverLayer);
            }

            this._updateAnimationTargets();

            // Create a copy
            this.currentStates = states.slice();
            this.markRedraw();

            if (!useHoverLayer && this.__inHover) {
                // Leave hover layer after states update and markRedraw.
                this._toggleHoverLayerFlag(false);
                // NOTE: avoid unexpected refresh when moving out from hover layer!!
                // Only clear from hover layer.
                this.__dirty &= ~REDRAW_BIT;
            }
        }
    }

    /**
     * Return if el.silent or any ancestor element has silent true.
     */
    isSilent() {
        let isSilent = this.silent;
        let ancestor = this.parent;
        while (!isSilent && ancestor) {
            if (ancestor.silent) {
                isSilent = true;
                break;
            }
            ancestor = ancestor.parent;
        }
        return isSilent;
    }

    /**
     * Update animation targets when reference is changed.
     */
    private _updateAnimationTargets() {
        for (let i = 0; i < this.animators.length; i++) {
            const animator = this.animators[i];
            if (animator.targetName) {
                animator.changeTarget((this as any)[animator.targetName]);
            }
        }
    }

    /**
     * Remove state
     * @param state State to remove
     */
    removeState(state: string) {
        const idx = indexOf(this.currentStates, state);
        if (idx >= 0) {
            const currentStates = this.currentStates.slice();
            currentStates.splice(idx, 1);
            this.useStates(currentStates);
        }
    }

    /**
     * Replace exists state.
     * @param oldState
     * @param newState
     * @param forceAdd If still add when even if replaced target not exists.
     */
    replaceState(oldState: string, newState: string, forceAdd: boolean) {
        const currentStates = this.currentStates.slice();
        const idx = indexOf(currentStates, oldState);
        const newStateExists = indexOf(currentStates, newState) >= 0;
        if (idx >= 0) {
            if (!newStateExists) {
                // Replace the old with the new one.
                currentStates[idx] = newState;
            }
            else {
                // Only remove the old one.
                currentStates.splice(idx, 1);
            }
        }
        else if (forceAdd && !newStateExists) {
            currentStates.push(newState);
        }
        this.useStates(currentStates);
    }

    /**
     * Toogle state.
     */
    toggleState(state: string, enable: boolean) {
        if (enable) {
            this.useState(state, true);
        }
        else {
            this.removeState(state);
        }
    }

    protected _mergeStates(states: ElementState[]) {
        const mergedState: ElementState = {};
        let mergedTextConfig: ElementTextConfig;
        for (let i = 0; i < states.length; i++) {
            const state = states[i];
            extend(mergedState, state);

            if (state.textConfig) {
                mergedTextConfig = mergedTextConfig || {};
                extend(mergedTextConfig, state.textConfig);
            }
        }
        if (mergedTextConfig) {
            mergedState.textConfig = mergedTextConfig;
        }

        return mergedState;
    }

    protected _applyStateObj(
        stateName: string,
        state: ElementState,
        normalState: ElementState,
        keepCurrentStates: boolean,
        transition: boolean,
        animationCfg: ElementAnimateConfig
    ) {
        const needsRestoreToNormal = !(state && keepCurrentStates);

        // TODO: Save current state to normal?
        // TODO: Animation
        if (state && state.textConfig) {
            // Inherit from current state or normal state.
            this.textConfig = extend(
                {},
                keepCurrentStates ? this.textConfig : normalState.textConfig
            );
            extend(this.textConfig, state.textConfig);
        }
        else if (needsRestoreToNormal) {
            if (normalState.textConfig) {   // Only restore if changed and saved.
                this.textConfig = normalState.textConfig;
            }
        }

        const transitionTarget: Dictionary<any> = {};
        let hasTransition = false;

        for (let i = 0; i < PRIMARY_STATES_KEYS.length; i++) {
            const key = PRIMARY_STATES_KEYS[i];
            const propNeedsTransition = transition && DEFAULT_ANIMATABLE_MAP[key];

            if (state && state[key] != null) {
                if (propNeedsTransition) {
                    hasTransition = true;
                    transitionTarget[key] = state[key];
                }
                else {
                    // Replace if it exist in target state
                    (this as any)[key] = state[key];
                }
            }
            else if (needsRestoreToNormal) {
                if (normalState[key] != null) {
                    if (propNeedsTransition) {
                        hasTransition = true;
                        transitionTarget[key] = normalState[key];
                    }
                    else {
                        // Restore to normal state
                        (this as any)[key] = normalState[key];
                    }
                }
            }
        }

        if (!transition) {
            // Keep the running animation to the new values after states changed.
            // Not simply stop animation. Or it may have jump effect.
            for (let i = 0; i < this.animators.length; i++) {
                const animator = this.animators[i];
                const targetName = animator.targetName;
                // Ignore loop animation
                if (!animator.getLoop()) {
                    animator.__changeFinalValue(targetName
                        ? ((state || normalState) as any)[targetName]
                        : (state || normalState)
                    );
                }
            }
        }

        if (hasTransition) {
            this._transitionState(
                stateName,
                transitionTarget as Props,
                animationCfg
            );
        }
    }

    /**
     * Component is some elements attached on this element for specific purpose.
     * Like clipPath, textContent
     */
    private _attachComponent(componentEl: Element) {
        if (componentEl.__zr && !componentEl.__hostTarget) {
            if (process.env.NODE_ENV !== 'production') {
                throw new Error('Text element has been added to zrender.');
            }
            return;
        }

        if (componentEl === this) {
            if (process.env.NODE_ENV !== 'production') {
                throw new Error('Recursive component attachment.');
            }
            return;
        }

        const zr = this.__zr;
        if (zr) {
            // Needs to add self to zrender. For rerender triggering, or animation.
            componentEl.addSelfToZr(zr);
        }

        componentEl.__zr = zr;
        componentEl.__hostTarget = this as unknown as Element;
    }

    private _detachComponent(componentEl: Element) {
        if (componentEl.__zr) {
            componentEl.removeSelfFromZr(componentEl.__zr);
        }

        componentEl.__zr = null;
        componentEl.__hostTarget = null;
    }

    /**
     * Get clip path
     */
    getClipPath() {
        return this._clipPath;
    }

    /**
     * Set clip path
     *
     * clipPath can't be shared between two elements.
     */
    setClipPath(clipPath: Path) {
        // Remove previous clip path
        if (this._clipPath && this._clipPath !== clipPath) {
            this.removeClipPath();
        }

        this._attachComponent(clipPath);

        this._clipPath = clipPath;
        this.markRedraw();
    }

    /**
     * Remove clip path
     */
    removeClipPath() {
        const clipPath = this._clipPath;
        if (clipPath) {
            this._detachComponent(clipPath);
            this._clipPath = null;
            this.markRedraw();
        }
    }

    /**
     * Get attached text content.
     */
    getTextContent(): ZRText {
        return this._textContent;
    }

    /**
     * Attach text on element
     */
    setTextContent(textEl: ZRText) {
        const previousTextContent = this._textContent;
        if (previousTextContent === textEl) {
            return;
        }
        // Remove previous textContent
        if (previousTextContent && previousTextContent !== textEl) {
            this.removeTextContent();
        }
        if (process.env.NODE_ENV !== 'production') {
            if (textEl.__zr && !textEl.__hostTarget) {
                throw new Error('Text element has been added to zrender.');
            }
        }

        textEl.innerTransformable = new Transformable();

        this._attachComponent(textEl);

        this._textContent = textEl;

        this.markRedraw();
    }

    /**
     * Set layout of attached text. Will merge with the previous.
     */
    setTextConfig(cfg: ElementTextConfig) {
        // TODO hide cfg property?
        if (!this.textConfig) {
            this.textConfig = {};
        }
        extend(this.textConfig, cfg);
        this.markRedraw();
    }

    /**
     * Remove text config
     */
    removeTextConfig() {
        this.textConfig = null;
        this.markRedraw();
    }

    /**
     * Remove attached text element.
     */
    removeTextContent() {
        const textEl = this._textContent;
        if (textEl) {
            textEl.innerTransformable = null;
            this._detachComponent(textEl);
            this._textContent = null;
            this._innerTextDefaultStyle = null;
            this.markRedraw();
        }
    }

    getTextGuideLine(): Polyline {
        return this._textGuide;
    }

    setTextGuideLine(guideLine: Polyline) {
        // Remove previous clip path
        if (this._textGuide && this._textGuide !== guideLine) {
            this.removeTextGuideLine();
        }

        this._attachComponent(guideLine);

        this._textGuide = guideLine;

        this.markRedraw();
    }

    removeTextGuideLine() {
        const textGuide = this._textGuide;
        if (textGuide) {
            this._detachComponent(textGuide);
            this._textGuide = null;
            this.markRedraw();
        }
    }
    /**
     * Mark element needs to be repainted
     */
    markRedraw() {
        this.__dirty |= REDRAW_BIT;
        const zr = this.__zr;
        if (zr) {
            if (this.__inHover) {
                zr.refreshHover();
            }
            else {
                zr.refresh();
            }
        }

        // Used as a clipPath or textContent
        if (this.__hostTarget) {
            this.__hostTarget.markRedraw();
        }
    }

    /**
     * Besides marking elements to be refreshed.
     * It will also invalid all cache and doing recalculate next frame.
     */
    dirty() {
        this.markRedraw();
    }

    private _toggleHoverLayerFlag(inHover: boolean) {
        this.__inHover = inHover;
        const textContent = this._textContent;
        const textGuide = this._textGuide;
        if (textContent) {
            textContent.__inHover = inHover;
        }
        if (textGuide) {
            textGuide.__inHover = inHover;
        }
    }

    /**
     * Add self from zrender instance.
     * Not recursively because it will be invoked when element added to storage.
     */
    addSelfToZr(zr: ZRenderType) {
        if (this.__zr === zr) {
            return;
        }

        this.__zr = zr;
        // 添加动画
        const animators = this.animators;
        if (animators) {
            for (let i = 0; i < animators.length; i++) {
                zr.animation.addAnimator(animators[i]);
            }
        }

        if (this._clipPath) {
            this._clipPath.addSelfToZr(zr);
        }
        if (this._textContent) {
            this._textContent.addSelfToZr(zr);
        }
        if (this._textGuide) {
            this._textGuide.addSelfToZr(zr);
        }
    }

    /**
     * Remove self from zrender instance.
     * Not recursively because it will be invoked when element added to storage.
     */
    removeSelfFromZr(zr: ZRenderType) {
        if (!this.__zr) {
            return;
        }

        this.__zr = null;
        // Remove animation
        const animators = this.animators;
        if (animators) {
            for (let i = 0; i < animators.length; i++) {
                zr.animation.removeAnimator(animators[i]);
            }
        }

        if (this._clipPath) {
            this._clipPath.removeSelfFromZr(zr);
        }
        if (this._textContent) {
            this._textContent.removeSelfFromZr(zr);
        }
        if (this._textGuide) {
            this._textGuide.removeSelfFromZr(zr);
        }
    }

    /**
     * 动画
     *
     * @param path The key to fetch value from object. Mostly style or shape.
     * @param loop Whether to loop animation.
     * @param allowDiscreteAnimation Whether to allow discrete animation
     * @example:
     *     el.animate('style', false)
     *         .when(1000, {x: 10} )
     *         .done(function(){ // Animation done })
     *         .start()
     */
    animate(key?: string, loop?: boolean, allowDiscreteAnimation?: boolean) {
        let target = key ? (this as any)[key] : this;

        if (process.env.NODE_ENV !== 'production') {
            if (!target) {
                logError(
                    'Property "'
                    + key
                    + '" is not existed in element '
                    + this.id
                );
                return;
            }
        }

        const animator = new Animator(target, loop, allowDiscreteAnimation);
        key && (animator.targetName = key);
        this.addAnimator(animator, key);
        return animator;
    }

    addAnimator(animator: Animator<any>, key: string): void {
        const zr = this.__zr;

        const el = this;

        animator.during(function () {
            el.updateDuringAnimation(key as string);
        }).done(function () {
            const animators = el.animators;
            // FIXME Animator will not be removed if use `Animator#stop` to stop animation
            const idx = indexOf(animators, animator);
            if (idx >= 0) {
                animators.splice(idx, 1);
            }
        });

        this.animators.push(animator);

        // If animate after added to the zrender
        if (zr) {
            zr.animation.addAnimator(animator);
        }

        // Wake up zrender to start the animation loop.
        zr && zr.wakeUp();
    }

    updateDuringAnimation(key: string) {
        this.markRedraw();
    }

    /**
     * 停止动画
     * @param {boolean} forwardToLast If move to last frame before stop
     */
    stopAnimation(scope?: string, forwardToLast?: boolean) {
        const animators = this.animators;
        const len = animators.length;
        const leftAnimators: Animator<any>[] = [];
        for (let i = 0; i < len; i++) {
            const animator = animators[i];
            if (!scope || scope === animator.scope) {
                animator.stop(forwardToLast);
            }
            else {
                leftAnimators.push(animator);
            }
        }
        this.animators = leftAnimators;

        return this;
    }

    /**
     * @param animationProps A map to specify which property to animate. If not specified, will animate all.
     * @example
     *  // Animate position
     *  el.animateTo({
     *      position: [10, 10]
     *  }, { done: () => { // done } })
     *
     *  // Animate shape, style and position in 100ms, delayed 100ms, with cubicOut easing
     *  el.animateTo({
     *      shape: {
     *          width: 500
     *      },
     *      style: {
     *          fill: 'red'
     *      }
     *      position: [10, 10]
     *  }, {
     *      duration: 100,
     *      delay: 100,
     *      easing: 'cubicOut',
     *      done: () => { // done }
     *  })
     */
    animateTo(target: Props, cfg?: ElementAnimateConfig, animationProps?: MapToType<Props, boolean>) {
        animateTo(this, target, cfg, animationProps);
    }

    /**
     * Animate from the target state to current state.
     * The params and the value are the same as `this.animateTo`.
     */

    // Overload definitions
    animateFrom(
        target: Props, cfg: ElementAnimateConfig, animationProps?: MapToType<Props, boolean>
    ) {
        animateTo(this, target, cfg, animationProps, true);
    }

    protected _transitionState(
        stateName: string, target: Props, cfg?: ElementAnimateConfig, animationProps?: MapToType<Props, boolean>
    ) {
        const animators = animateTo(this, target, cfg, animationProps);
        for (let i = 0; i < animators.length; i++) {
            animators[i].__fromStateTransition = stateName;
        }
    }

    /**
     * Interface of getting the minimum bounding box.
     */
    getBoundingRect(): BoundingRect {
        return null;
    }

    getPaintRect(): BoundingRect {
        return null;
    }

    /**
     * The string value of `textPosition` needs to be calculated to a real postion.
     * For example, `'inside'` is calculated to `[rect.width/2, rect.height/2]`
     * by default. See `contain/text.js#calculateTextPosition` for more details.
     * But some coutom shapes like "pin", "flag" have center that is not exactly
     * `[width/2, height/2]`. So we provide this hook to customize the calculation
     * for those shapes. It will be called if the `style.textPosition` is a string.
     * @param {Obejct} [out] Prepared out object. If not provided, this method should
     *        be responsible for creating one.
     * @param {module:zrender/graphic/Style} style
     * @param {Object} rect {x, y, width, height}
     * @return {Obejct} out The same as the input out.
     *         {
     *             x: number. mandatory.
     *             y: number. mandatory.
     *             align: string. optional. use style.textAlign by default.
     *             verticalAlign: string. optional. use style.textVerticalAlign by default.
     *         }
     */
    calculateTextPosition: ElementCalculateTextPosition;

    protected static initDefaultProps = (function () {
        const elProto = Element.prototype;
        elProto.type = 'element';
        elProto.name = '';

        elProto.ignore =
        elProto.silent =
        elProto.isGroup =
        elProto.draggable =
        elProto.dragging =
        elProto.ignoreClip =
        elProto.__inHover = false;

        elProto.__dirty = REDRAW_BIT;


        const logs: Dictionary<boolean> = {};
        function logDeprecatedError(key: string, xKey: string, yKey: string) {
            if (!logs[key + xKey + yKey]) {
                console.warn(`DEPRECATED: '${key}' has been deprecated. use '${xKey}', '${yKey}' instead`);
                logs[key + xKey + yKey] = true;
            }
        }
        // Legacy transform properties. position and scale
        function createLegacyProperty(
            key: string,
            privateKey: string,
            xKey: string,
            yKey: string
        ) {
            Object.defineProperty(elProto, key, {
                get() {
                    if (process.env.NODE_ENV !== 'production') {
                        logDeprecatedError(key, xKey, yKey);
                    }
                    if (!this[privateKey]) {
                        const pos: number[] = this[privateKey] = [];
                        enhanceArray(this, pos);
                    }
                    return this[privateKey];
                },
                set(pos: number[]) {
                    if (process.env.NODE_ENV !== 'production') {
                        logDeprecatedError(key, xKey, yKey);
                    }
                    this[xKey] = pos[0];
                    this[yKey] = pos[1];
                    this[privateKey] = pos;
                    enhanceArray(this, pos);
                }
            });
            function enhanceArray(self: any, pos: number[]) {
                Object.defineProperty(pos, 0, {
                    get() {
                        return self[xKey];
                    },
                    set(val: number) {
                        self[xKey] = val;
                    }
                });
                Object.defineProperty(pos, 1, {
                    get() {
                        return self[yKey];
                    },
                    set(val: number) {
                        self[yKey] = val;
                    }
                });
            }
        }
        if (Object.defineProperty
            // Just don't support ie8
            // && (!(env as any).browser.ie || (env as any).browser.version > 8)
        ) {
            createLegacyProperty('position', '_legacyPos', 'x', 'y');
            createLegacyProperty('scale', '_legacyScale', 'scaleX', 'scaleY');
            createLegacyProperty('origin', '_legacyOrigin', 'originX', 'originY');
        }
    })()
}

mixin(Element, Eventful);
mixin(Element, Transformable);

function animateTo<T>(
    animatable: Element<T>,
    target: Dictionary<any>,
    cfg: ElementAnimateConfig,
    animationProps: Dictionary<any>,
    reverse?: boolean
) {
    cfg = cfg || {};
    const animators: Animator<any>[] = [];
    animateToShallow(
        animatable,
        '',
        animatable,
        target,
        cfg,
        animationProps,
        animators,
        reverse
    );

    let finishCount = animators.length;
    let doneHappened = false;
    const cfgDone = cfg.done;
    const cfgAborted = cfg.aborted;

    const doneCb = () => {
        doneHappened = true;
        finishCount--;
        if (finishCount <= 0) {
            doneHappened
                ? (cfgDone && cfgDone())
                : (cfgAborted && cfgAborted());
        }
    };

    const abortedCb = () => {
        finishCount--;
        if (finishCount <= 0) {
            doneHappened
                ? (cfgDone && cfgDone())
                : (cfgAborted && cfgAborted());
        }
    };

    // No animators. This should be checked before animators[i].start(),
    // because 'done' may be executed immediately if no need to animate.
    if (!finishCount) {
        cfgDone && cfgDone();
    }

    // Adding during callback to the first animator
    if (animators.length > 0 && cfg.during) {
        // TODO If there are two animators in animateTo, and the first one is stopped by other animator.
        animators[0].during((target, percent) => {
            cfg.during(percent);
        });
    }

    // Start after all animators created
    // Incase any animator is done immediately when all animation properties are not changed
    for (let i = 0; i < animators.length; i++) {
        const animator = animators[i];
        if (doneCb) {
            animator.done(doneCb);
        }
        if (abortedCb) {
            animator.aborted(abortedCb);
        }
        if (cfg.force) {
            animator.duration(cfg.duration);
        }
        animator.start(cfg.easing);
    }

    return animators;
}

function copyArrShallow(source: number[], target: number[], len: number) {
    for (let i = 0; i < len; i++) {
        source[i] = target[i];
    }
}

function is2DArray(value: any[]): value is number[][] {
    return isArrayLike(value[0]);
}

function copyValue(target: Dictionary<any>, source: Dictionary<any>, key: string) {
    if (isArrayLike(source[key])) {
        if (!isArrayLike(target[key])) {
            target[key] = [];
        }

        if (isTypedArray(source[key])) {
            const len = source[key].length;
            if (target[key].length !== len) {
                target[key] = new (source[key].constructor)(len);
                copyArrShallow(target[key], source[key], len);
            }
        }
        else {
            const sourceArr = source[key] as any[];
            const targetArr = target[key] as any[];

            const len0 = sourceArr.length;
            if (is2DArray(sourceArr)) {
                // NOTE: each item should have same length
                const len1 = sourceArr[0].length;

                for (let i = 0; i < len0; i++) {
                    if (!targetArr[i]) {
                        targetArr[i] = Array.prototype.slice.call(sourceArr[i]);
                    }
                    else {
                        copyArrShallow(targetArr[i], sourceArr[i], len1);
                    }
                }
            }
            else {
                copyArrShallow(targetArr, sourceArr, len0);
            }

            targetArr.length = sourceArr.length;
        }
    }
    else {
        target[key] = source[key];
    }
}

function isValueSame(val1: any, val2: any) {
    return val1 === val2
        // Only check 1 dimension array
        || isArrayLike(val1) && isArrayLike(val2) && is1DArraySame(val1, val2);
}

function is1DArraySame(arr0: ArrayLike<number>, arr1: ArrayLike<number>) {
    const len = arr0.length;
    if (len !== arr1.length) {
        return false;
    }
    for (let i = 0; i < len; i++) {
        if (arr0[i] !== arr1[i]) {
            return false;
        }
    }
    return true;
}

function animateToShallow<T>(
    animatable: Element<T>,
    topKey: string,
    animateObj: Dictionary<any>,
    target: Dictionary<any>,
    cfg: ElementAnimateConfig,
    animationProps: Dictionary<any> | true,
    animators: Animator<any>[],
    reverse: boolean    // If `true`, animate from the `target` to current state.
) {
    const targetKeys = keys(target);
    const duration = cfg.duration;
    const delay = cfg.delay;
    const additive = cfg.additive;
    const setToFinal = cfg.setToFinal;
    const animateAll = !isObject(animationProps);
    // Find last animator animating same prop.
    const existsAnimators = animatable.animators;

    let animationKeys: string[] = [];
    for (let k = 0; k < targetKeys.length; k++) {
        const innerKey = targetKeys[k] as string;
        const targetVal = target[innerKey];

        if (
            targetVal != null && animateObj[innerKey] != null
            && (animateAll || (animationProps as Dictionary<any>)[innerKey])
        ) {
            if (isObject(targetVal)
                && !isArrayLike(targetVal)
                && !isGradientObject(targetVal)
            ) {
                if (topKey) {
                    // logError('Only support 1 depth nest object animation.');
                    // Assign directly.
                    // TODO richText?
                    if (!reverse) {
                        animateObj[innerKey] = targetVal;
                        animatable.updateDuringAnimation(topKey);
                    }
                    continue;
                }
                animateToShallow(
                    animatable,
                    innerKey,
                    animateObj[innerKey],
                    targetVal,
                    cfg,
                    animationProps && (animationProps as Dictionary<any>)[innerKey],
                    animators,
                    reverse
                );
            }
            else {
                animationKeys.push(innerKey);
            }
        }
        else if (!reverse) {
            // Assign target value directly.
            animateObj[innerKey] = targetVal;
            animatable.updateDuringAnimation(topKey);
            // Previous animation will be stopped on the changed keys.
            // So direct assign is also included.
            animationKeys.push(innerKey);
        }
    }

    let keyLen = animationKeys.length;
    // Stop previous animations on the same property.
    if (!additive && keyLen) {
        // Stop exists animation on specific tracks. Only one animator available for each property.
        // TODO Should invoke previous animation callback?
        for (let i = 0; i < existsAnimators.length; i++) {
            const animator = existsAnimators[i];
            if (animator.targetName === topKey) {
                const allAborted = animator.stopTracks(animationKeys);
                if (allAborted) {   // This animator can't be used.
                    const idx = indexOf(existsAnimators, animator);
                    existsAnimators.splice(idx, 1);
                }
            }
        }
    }

    // Ignore values not changed.
    // NOTE: Must filter it after previous animation stopped
    // and make sure the value to compare is using initial frame if animation is not started yet when setToFinal is used.
    if (!cfg.force) {
        animationKeys = filter(animationKeys, key => !isValueSame(target[key], animateObj[key]));
        keyLen = animationKeys.length;
    }

    if (keyLen > 0
        // cfg.force is mainly for keep invoking onframe and ondone callback even if animation is not necessary.
        // So if there is already has animators. There is no need to create another animator if not necessary.
        // Or it will always add one more with empty target.
        || (cfg.force && !animators.length)
    ) {
        let revertedSource: Dictionary<any>;
        let reversedTarget: Dictionary<any>;
        let sourceClone: Dictionary<any>;
        if (reverse) {
            reversedTarget = {};
            if (setToFinal) {
                revertedSource = {};
            }
            for (let i = 0; i < keyLen; i++) {
                const innerKey = animationKeys[i];
                reversedTarget[innerKey] = animateObj[innerKey];
                if (setToFinal) {
                    revertedSource[innerKey] = target[innerKey];
                }
                else {
                    // The usage of "animateFrom" expects that the element props has been updated dirctly to
                    // "final" values outside, and input the "from" values here (i.e., in variable `target` here).
                    // So here we assign the "from" values directly to element here (rather that in the next frame)
                    // to prevent the "final" values from being read in any other places (like other running
                    // animator during callbacks).
                    // But if `setToFinal: true` this feature can not be satisfied.
                    animateObj[innerKey] = target[innerKey];
                }
            }
        }
        else if (setToFinal) {
            sourceClone = {};
            for (let i = 0; i < keyLen; i++) {
                const innerKey = animationKeys[i];
                // NOTE: Must clone source after the stopTracks. The property may be modified in stopTracks.
                sourceClone[innerKey] = cloneValue(animateObj[innerKey]);
                // Use copy, not change the original reference
                // Copy from target to source.
                copyValue(animateObj, target, innerKey);
            }
        }

        const animator = new Animator(animateObj, false, false, additive ? filter(
            // Use key string instead object reference because ref may be changed.
            existsAnimators, animator => animator.targetName === topKey
        ) : null);

        animator.targetName = topKey;
        if (cfg.scope) {
            animator.scope = cfg.scope;
        }

        if (setToFinal && revertedSource) {
            animator.whenWithKeys(0, revertedSource, animationKeys);
        }
        if (sourceClone) {
            animator.whenWithKeys(0, sourceClone, animationKeys);
        }

        animator.whenWithKeys(
            duration == null ? 500 : duration,
            reverse ? reversedTarget : target,
            animationKeys
        ).delay(delay || 0);

        animatable.addAnimator(animator, topKey);
        animators.push(animator);
    }
}


export default Element;
