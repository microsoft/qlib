/**
 * Group是一个容器，可以插入子节点，Group的变换也会被应用到子节点上
 * @module zrender/graphic/Group
 * @example
 *     const Group = require('zrender/graphic/Group');
 *     const Circle = require('zrender/graphic/shape/Circle');
 *     const g = new Group();
 *     g.position[0] = 100;
 *     g.position[1] = 100;
 *     g.add(new Circle({
 *         style: {
 *             x: 100,
 *             y: 100,
 *             r: 20,
 *         }
 *     }));
 *     zr.add(g);
 */

import * as zrUtil from '../core/util';
import Element, { ElementProps } from '../Element';
import BoundingRect from '../core/BoundingRect';
import { MatrixArray } from '../core/matrix';
import Displayable from './Displayable';
import { ZRenderType } from '../zrender';

export interface GroupProps extends ElementProps {
}

class Group extends Element<GroupProps> {

    readonly isGroup = true

    private _children: Element[] = []


    constructor(opts?: GroupProps) {
        super();

        this.attr(opts);
    }

    /**
     * Get children reference.
     */
    childrenRef() {
        return this._children;
    }

    /**
     * Get children copy.
     */
    children() {
        return this._children.slice();
    }

    /**
     * 获取指定 index 的儿子节点
     */
    childAt(idx: number): Element {
        return this._children[idx];
    }

    /**
     * 获取指定名字的儿子节点
     */
    childOfName(name: string): Element {
        const children = this._children;
        for (let i = 0; i < children.length; i++) {
            if (children[i].name === name) {
                return children[i];
            }
        }
    }

    childCount(): number {
        return this._children.length;
    }

    /**
     * 添加子节点到最后
     */
    add(child: Element): Group {
        if (child) {
            if (child !== this && child.parent !== this) {
                this._children.push(child);
                this._doAdd(child);
            }
            if (process.env.NODE_ENV !== 'production') {
                if (child.__hostTarget) {
                    throw 'This elemenet has been used as an attachment';
                }
            }
        }

        return this;
    }

    /**
     * 添加子节点在 nextSibling 之前
     */
    addBefore(child: Element, nextSibling: Element) {
        if (child && child !== this && child.parent !== this
            && nextSibling && nextSibling.parent === this) {

            const children = this._children;
            const idx = children.indexOf(nextSibling);

            if (idx >= 0) {
                children.splice(idx, 0, child);
                this._doAdd(child);
            }
        }

        return this;
    }

    replace(oldChild: Element, newChild: Element) {
        const idx = zrUtil.indexOf(this._children, oldChild);
        if (idx >= 0) {
            this.replaceAt(newChild, idx);
        }
        return this;
    }

    replaceAt(child: Element, index: number) {
        const children = this._children;
        const old = children[index];

        if (child && child !== this && child.parent !== this && child !== old) {
            children[index] = child;

            old.parent = null;
            const zr = this.__zr;
            if (zr) {
                old.removeSelfFromZr(zr);
            }

            this._doAdd(child);
        }

        return this;
    }

    _doAdd(child: Element) {
        if (child.parent) {
            // Parent must be a group
            (child.parent as Group).remove(child);
        }

        child.parent = this;

        const zr = this.__zr;
        if (zr && zr !== (child as Group).__zr) {    // Only group has __storage

            child.addSelfToZr(zr);
        }

        zr && zr.refresh();
    }

    /**
     * Remove child
     * @param child
     */
    remove(child: Element) {
        const zr = this.__zr;
        const children = this._children;

        const idx = zrUtil.indexOf(children, child);
        if (idx < 0) {
            return this;
        }
        children.splice(idx, 1);

        child.parent = null;

        if (zr) {

            child.removeSelfFromZr(zr);
        }

        zr && zr.refresh();

        return this;
    }

    /**
     * Remove all children
     */
    removeAll() {
        const children = this._children;
        const zr = this.__zr;
        for (let i = 0; i < children.length; i++) {
            const child = children[i];
            if (zr) {
                child.removeSelfFromZr(zr);
            }
            child.parent = null;
        }
        children.length = 0;

        return this;
    }

    /**
     * 遍历所有子节点
     */
    eachChild<Context>(
        cb: (this: Context, el: Element, index?: number) => void,
        context?: Context
    ) {
        const children = this._children;
        for (let i = 0; i < children.length; i++) {
            const child = children[i];
            cb.call(context, child, i);
        }
        return this;
    }

    /**
     * Visit all descendants.
     * Return false in callback to stop visit descendants of current node
     */
    // TODO Group itself should also invoke the callback.
    traverse<T>(
        cb: (this: T, el: Element) => boolean | void,
        context?: T
    ) {
        for (let i = 0; i < this._children.length; i++) {
            const child = this._children[i];
            const stopped = cb.call(context, child);

            if (child.isGroup && !stopped) {
                child.traverse(cb, context);
            }
        }
        return this;
    }

    addSelfToZr(zr: ZRenderType) {
        super.addSelfToZr(zr);
        for (let i = 0; i < this._children.length; i++) {
            const child = this._children[i];
            child.addSelfToZr(zr);
        }
    }

    removeSelfFromZr(zr: ZRenderType) {
        super.removeSelfFromZr(zr);
        for (let i = 0; i < this._children.length; i++) {
            const child = this._children[i];
            child.removeSelfFromZr(zr);
        }
    }

    getBoundingRect(includeChildren?: Element[]): BoundingRect {
        // TODO Caching
        const tmpRect = new BoundingRect(0, 0, 0, 0);
        const children = includeChildren || this._children;
        const tmpMat: MatrixArray = [];
        let rect = null;

        for (let i = 0; i < children.length; i++) {
            const child = children[i];
            // TODO invisible?
            if (child.ignore || (child as Displayable).invisible) {
                continue;
            }

            const childRect = child.getBoundingRect();
            const transform = child.getLocalTransform(tmpMat);
            // TODO
            // The boundingRect cacluated by transforming original
            // rect may be bigger than the actual bundingRect when rotation
            // is used. (Consider a circle rotated aginst its center, where
            // the actual boundingRect should be the same as that not be
            // rotated.) But we can not find better approach to calculate
            // actual boundingRect yet, considering performance.
            if (transform) {
                BoundingRect.applyTransform(tmpRect, childRect, transform);
                rect = rect || tmpRect.clone();
                rect.union(tmpRect);
            }
            else {
                rect = rect || childRect.clone();
                rect.union(childRect);
            }
        }
        return rect || tmpRect;
    }
}

Group.prototype.type = 'group';
// Storage will use childrenRef to get children to render.
export interface GroupLike extends Element {
    childrenRef(): Element[]
}

export default Group;