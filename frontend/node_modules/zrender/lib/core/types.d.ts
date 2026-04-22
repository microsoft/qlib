export declare type Dictionary<T> = {
    [key: string]: T;
};
export declare type ArrayLike<T> = {
    [key: number]: T;
    length: number;
};
export declare type ImageLike = HTMLImageElement | HTMLCanvasElement | HTMLVideoElement;
export declare type TextVerticalAlign = 'top' | 'middle' | 'bottom';
export declare type TextAlign = 'left' | 'center' | 'right';
export declare type FontWeight = 'normal' | 'bold' | 'bolder' | 'lighter' | number;
export declare type FontStyle = 'normal' | 'italic' | 'oblique';
export declare type BuiltinTextPosition = 'left' | 'right' | 'top' | 'bottom' | 'inside' | 'insideLeft' | 'insideRight' | 'insideTop' | 'insideBottom' | 'insideTopLeft' | 'insideTopRight' | 'insideBottomLeft' | 'insideBottomRight';
export declare type WXCanvasRenderingContext = CanvasRenderingContext2D & {
    draw: () => void;
};
export declare type ZRCanvasRenderingContext = CanvasRenderingContext2D & {
    dpr: number;
    __attrCachedBy: boolean | number;
};
declare type ZREventProperties = {
    zrX: number;
    zrY: number;
    zrDelta: number;
    zrEventControl: 'no_globalout' | 'only_globalout';
    zrByTouch: boolean;
};
export declare type ZRRawMouseEvent = MouseEvent & ZREventProperties;
export declare type ZRRawTouchEvent = TouchEvent & ZREventProperties;
export declare type ZRRawPointerEvent = TouchEvent & ZREventProperties;
export declare type ZRRawEvent = ZRRawMouseEvent | ZRRawTouchEvent | ZRRawPointerEvent;
export declare type ZRPinchEvent = ZRRawEvent & {
    pinchScale: number;
    pinchX: number;
    pinchY: number;
    gestureEvent: string;
};
export declare type ElementEventName = 'click' | 'dblclick' | 'mousewheel' | 'mouseout' | 'mouseover' | 'mouseup' | 'mousedown' | 'mousemove' | 'contextmenu' | 'drag' | 'dragstart' | 'dragend' | 'dragenter' | 'dragleave' | 'dragover' | 'drop' | 'globalout';
export declare type ElementEventNameWithOn = 'onclick' | 'ondblclick' | 'onmousewheel' | 'onmouseout' | 'onmouseup' | 'onmousedown' | 'onmousemove' | 'oncontextmenu' | 'ondrag' | 'ondragstart' | 'ondragend' | 'ondragenter' | 'ondragleave' | 'ondragover' | 'ondrop';
export declare type RenderedEvent = {
    elapsedTime: number;
};
export declare type PropType<TObj, TProp extends keyof TObj> = TObj[TProp];
export declare type AllPropTypes<T> = PropType<T, keyof T>;
export declare type FunctionPropertyNames<T> = {
    [K in keyof T]: T[K] extends Function ? K : never;
}[keyof T];
export declare type MapToType<T extends Dictionary<any>, S> = {
    [P in keyof T]: T[P] extends Dictionary<any> ? MapToType<T[P], S> : S;
};
export declare type KeyOfDistributive<T> = T extends unknown ? keyof T : never;
export declare type WithThisType<Func extends (...args: any) => any, This> = (this: This, ...args: Parameters<Func>) => ReturnType<Func>;
export {};
