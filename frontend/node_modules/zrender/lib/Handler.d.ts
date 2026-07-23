import Eventful from './core/Eventful';
import Displayable from './graphic/Displayable';
import { PainterBase } from './PainterBase';
import { HandlerProxyInterface } from './dom/HandlerProxy';
import { ZRRawEvent, ElementEventName } from './core/types';
import Storage from './Storage';
import Element from './Element';
declare class HoveredResult {
    x: number;
    y: number;
    target: Displayable;
    topTarget: Displayable;
    constructor(x?: number, y?: number);
}
declare type HandlerName = 'click' | 'dblclick' | 'mousewheel' | 'mouseout' | 'mouseup' | 'mousedown' | 'mousemove' | 'contextmenu';
declare class Handler extends Eventful {
    storage: Storage;
    painter: PainterBase;
    painterRoot: HTMLElement;
    proxy: HandlerProxyInterface;
    private _hovered;
    private _gestureMgr;
    private _draggingMgr;
    private _pointerSize;
    _downEl: Element;
    _upEl: Element;
    _downPoint: [number, number];
    constructor(storage: Storage, painter: PainterBase, proxy: HandlerProxyInterface, painterRoot: HTMLElement, pointerSize: number);
    setHandlerProxy(proxy: HandlerProxyInterface): void;
    mousemove(event: ZRRawEvent): void;
    mouseout(event: ZRRawEvent): void;
    resize(): void;
    dispatch(eventName: HandlerName, eventArgs?: any): void;
    dispose(): void;
    setCursorStyle(cursorStyle: string): void;
    dispatchToElement(targetInfo: {
        target?: Element;
        topTarget?: Element;
    }, eventName: ElementEventName, event: ZRRawEvent): void;
    findHover(x: number, y: number, exclude?: Displayable): HoveredResult;
    processGesture(event: ZRRawEvent, stage?: 'start' | 'end' | 'change'): void;
    click: (event: ZRRawEvent) => void;
    mousedown: (event: ZRRawEvent) => void;
    mouseup: (event: ZRRawEvent) => void;
    mousewheel: (event: ZRRawEvent) => void;
    dblclick: (event: ZRRawEvent) => void;
    contextmenu: (event: ZRRawEvent) => void;
}
export default Handler;
