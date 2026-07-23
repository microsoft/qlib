export type Dictionary<T> = {
    [key: string]: T
}

/**
 * Not readonly ArrayLike
 * Include Array, TypedArray
 */
export type ArrayLike<T> = {
    [key: number]: T
    length: number
}

export type ImageLike = HTMLImageElement | HTMLCanvasElement | HTMLVideoElement

// subset of CanvasTextBaseline
export type TextVerticalAlign = 'top' | 'middle' | 'bottom'
    // | 'center' // DEPRECATED

// TODO: Have not support 'start', 'end' yet.
// subset of CanvasTextAlign
export type TextAlign = 'left' | 'center' | 'right'
    // | 'middle' // DEPRECATED

export type FontWeight = 'normal' | 'bold' | 'bolder' | 'lighter' | number;
export type FontStyle = 'normal' | 'italic' | 'oblique';

export type BuiltinTextPosition = 'left' | 'right' | 'top' | 'bottom' | 'inside'
    | 'insideLeft' | 'insideRight' | 'insideTop' | 'insideBottom'
    | 'insideTopLeft' | 'insideTopRight'| 'insideBottomLeft' | 'insideBottomRight';

export type WXCanvasRenderingContext = CanvasRenderingContext2D & {
    draw: () => void
};

export type ZRCanvasRenderingContext = CanvasRenderingContext2D & {
    dpr: number
    __attrCachedBy: boolean | number
}

// Properties zrender will extended to the raw event
type ZREventProperties = {
    zrX: number
    zrY: number
    zrDelta: number

    // 'no_globalout' means: do not trigger "globalout" event to zr user.
    // 'only_globalout" means: only trigger "globalout" event, but do not
    //     trigger other event to zr user.
    zrEventControl: 'no_globalout' | 'only_globalout'

    zrByTouch: boolean
}

export type ZRRawMouseEvent = MouseEvent & ZREventProperties
export type ZRRawTouchEvent = TouchEvent & ZREventProperties
export type ZRRawPointerEvent = TouchEvent & ZREventProperties

export type ZRRawEvent = ZRRawMouseEvent | ZRRawTouchEvent | ZRRawPointerEvent

export type ZRPinchEvent = ZRRawEvent & {
    pinchScale: number
    pinchX: number
    pinchY: number
    gestureEvent: string
}

export type ElementEventName = 'click' | 'dblclick' | 'mousewheel' | 'mouseout' |
    'mouseover' | 'mouseup' | 'mousedown' | 'mousemove' | 'contextmenu' |
    'drag' | 'dragstart' | 'dragend' | 'dragenter' | 'dragleave' | 'dragover' | 'drop' | 'globalout';

export type ElementEventNameWithOn = 'onclick' | 'ondblclick' | 'onmousewheel' | 'onmouseout' |
    'onmouseup' | 'onmousedown' | 'onmousemove' | 'oncontextmenu' |
    'ondrag' | 'ondragstart' | 'ondragend' | 'ondragenter' | 'ondragleave' | 'ondragover' | 'ondrop';

export type RenderedEvent = {
    elapsedTime: number
};

// Useful type methods
export type PropType<TObj, TProp extends keyof TObj> = TObj[TProp];

export type AllPropTypes<T> = PropType<T, keyof T>

export type FunctionPropertyNames<T> = {[K in keyof T]: T[K] extends Function ? K : never}[keyof T];

export type MapToType<T extends Dictionary<any>, S> = {
    [P in keyof T]: T[P] extends Dictionary<any> ? MapToType<T[P], S> : S
}

// See https://www.staging-typescript.org/docs/handbook/advanced-types.html#distributive-conditional-types
// For the case:
// `keyof A | B` does not equals to `Keyof A | Keyof B`
// KeyOfDistributive<A | B> equals to `KeyOfDistributive<A> | KeyOfDistributive<B>`
export type KeyOfDistributive<T> = T extends unknown ? keyof T : never;

export type WithThisType<Func extends (...args: any) => any, This> =
    (this: This, ...args: Parameters<Func>) => ReturnType<Func>;
