import Element, { ElementProps } from '../Element';
import BoundingRect from '../core/BoundingRect';
import { ZRenderType } from '../zrender';
export interface GroupProps extends ElementProps {
}
declare class Group extends Element<GroupProps> {
    readonly isGroup = true;
    private _children;
    constructor(opts?: GroupProps);
    childrenRef(): Element<ElementProps>[];
    children(): Element<ElementProps>[];
    childAt(idx: number): Element;
    childOfName(name: string): Element;
    childCount(): number;
    add(child: Element): Group;
    addBefore(child: Element, nextSibling: Element): this;
    replace(oldChild: Element, newChild: Element): this;
    replaceAt(child: Element, index: number): this;
    _doAdd(child: Element): void;
    remove(child: Element): this;
    removeAll(): this;
    eachChild<Context>(cb: (this: Context, el: Element, index?: number) => void, context?: Context): this;
    traverse<T>(cb: (this: T, el: Element) => boolean | void, context?: T): this;
    addSelfToZr(zr: ZRenderType): void;
    removeSelfFromZr(zr: ZRenderType): void;
    getBoundingRect(includeChildren?: Element[]): BoundingRect;
}
export interface GroupLike extends Element {
    childrenRef(): Element[];
}
export default Group;
