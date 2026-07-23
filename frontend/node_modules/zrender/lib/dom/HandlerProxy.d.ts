import Eventful from '../core/Eventful';
import { VectorArray } from '../core/vector';
import Handler from '../Handler';
export default class HandlerDomProxy extends Eventful {
    dom: HTMLElement;
    painterRoot: HTMLElement;
    handler: Handler;
    private _localHandlerScope;
    private _globalHandlerScope;
    __lastTouchMoment: Date;
    __pointerCapturing: boolean;
    __mayPointerCapture: VectorArray;
    constructor(dom: HTMLElement, painterRoot: HTMLElement);
    dispose(): void;
    setCursor(cursorStyle: string): void;
    __togglePointerCapture(isPointerCapturing?: boolean): void;
}
export interface HandlerProxyInterface extends Eventful {
    handler: Handler;
    dispose: () => void;
    setCursor: (cursorStyle?: string) => void;
}
