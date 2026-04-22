import Eventful from './Eventful';
import { ZRRawEvent } from './types';
declare type FirefoxMouseEvent = {
    layerX: number;
    layerY: number;
};
export declare function clientToLocal(el: HTMLElement, e: ZRRawEvent | FirefoxMouseEvent | Touch, out: {
    zrX?: number;
    zrY?: number;
}, calculate?: boolean): {
    zrX?: number;
    zrY?: number;
};
export declare function getNativeEvent(e: ZRRawEvent): ZRRawEvent;
export declare function normalizeEvent(el: HTMLElement, e: ZRRawEvent, calculate?: boolean): ZRRawEvent;
declare type AddEventListenerParams = Parameters<typeof HTMLElement.prototype.addEventListener>;
declare type RemoveEventListenerParams = Parameters<typeof HTMLElement.prototype.removeEventListener>;
export declare function addEventListener(el: HTMLElement | HTMLDocument, name: AddEventListenerParams[0], handler: AddEventListenerParams[1], opt?: AddEventListenerParams[2]): void;
export declare function removeEventListener(el: HTMLElement | HTMLDocument, name: RemoveEventListenerParams[0], handler: RemoveEventListenerParams[1], opt: RemoveEventListenerParams[2]): void;
export declare const stop: (e: MouseEvent | TouchEvent | PointerEvent) => void;
export declare function isMiddleOrRightButtonOnMouseUpDown(e: {
    which: number;
}): boolean;
export { Eventful as Dispatcher };
