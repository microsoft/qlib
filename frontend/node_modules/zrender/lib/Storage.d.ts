import Element from './Element';
import Displayable from './graphic/Displayable';
declare function shapeCompareFunc(a: Displayable, b: Displayable): number;
export default class Storage {
    private _roots;
    private _displayList;
    private _displayListLen;
    traverse<T>(cb: (this: T, el: Element) => void, context?: T): void;
    getDisplayList(update?: boolean, includeIgnore?: boolean): Displayable[];
    updateDisplayList(includeIgnore?: boolean): void;
    private _updateAndAddDisplayable;
    addRoot(el: Element): void;
    delRoot(el: Element | Element[]): void;
    delAllRoots(): void;
    getRoots(): Element<import("./Element").ElementProps>[];
    dispose(): void;
    displayableSortFunc: typeof shapeCompareFunc;
}
export {};
