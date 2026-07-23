import Handler from '../Handler';
import Element, { ElementEvent } from '../Element';
export default class Draggable {
    handler: Handler;
    _draggingTarget: Element;
    _dropTarget: Element;
    _x: number;
    _y: number;
    constructor(handler: Handler);
    _dragStart(e: ElementEvent): void;
    _drag(e: ElementEvent): void;
    _dragEnd(e: ElementEvent): void;
}
