/**
 * Displayable for incremental rendering. It will be rendered in a separate layer
 * IncrementalDisplay have two main methods. `clearDisplayables` and `addDisplayables`
 * addDisplayables will render the added displayables incremetally.
 *
 * It use a notClear flag to tell the painter don't clear the layer if it's the first element.
 *
 * It's not available for SVG rendering.
 */
import Displayble from './Displayable';
import BoundingRect from '../core/BoundingRect';
import { MatrixArray } from '../core/matrix';
import Group from './Group';

const m: MatrixArray = [];
// TODO Style override ?

export default class IncrementalDisplayable extends Displayble {

    notClear: boolean = true

    incremental = true

    private _displayables: Displayble[] = []
    private _temporaryDisplayables: Displayble[] = []

    private _cursor = 0

    traverse<T>(
        cb: (this: T, el: this) => void,
        context: T
    ) {
        cb.call(context, this);
    }

    useStyle() {
        // Use an empty style
        // PENDING
        this.style = {};
    }

    // getCurrentCursor / updateCursorAfterBrush
    // is used in graphic.ts. It's not provided for developers
    getCursor() {
        return this._cursor;
    }
    // Update cursor after brush.
    innerAfterBrush() {
        this._cursor = this._displayables.length;
    }

    clearDisplaybles() {
        this._displayables = [];
        this._temporaryDisplayables = [];
        this._cursor = 0;
        this.markRedraw();

        this.notClear = false;
    }

    clearTemporalDisplayables() {
        this._temporaryDisplayables = [];
    }

    addDisplayable(displayable: Displayble, notPersistent?: boolean) {
        if (notPersistent) {
            this._temporaryDisplayables.push(displayable);
        }
        else {
            this._displayables.push(displayable);
        }
        this.markRedraw();
    }

    addDisplayables(displayables: Displayble[], notPersistent?: boolean) {
        notPersistent = notPersistent || false;
        for (let i = 0; i < displayables.length; i++) {
            this.addDisplayable(displayables[i], notPersistent);
        }
    }

    getDisplayables(): Displayble[] {
        return this._displayables;
    }

    getTemporalDisplayables(): Displayble[] {
        return this._temporaryDisplayables;
    }

    eachPendingDisplayable(cb: (displayable: Displayble) => void) {
        for (let i = this._cursor; i < this._displayables.length; i++) {
            cb && cb(this._displayables[i]);
        }
        for (let i = 0; i < this._temporaryDisplayables.length; i++) {
            cb && cb(this._temporaryDisplayables[i]);
        }
    }

    update() {
        this.updateTransform();
        for (let i = this._cursor; i < this._displayables.length; i++) {
            const displayable = this._displayables[i];
            // PENDING
            displayable.parent = this as unknown as Group;
            displayable.update();
            displayable.parent = null;
        }
        for (let i = 0; i < this._temporaryDisplayables.length; i++) {
            const displayable = this._temporaryDisplayables[i];
            // PENDING
            displayable.parent = this as unknown as Group;
            displayable.update();
            displayable.parent = null;
        }
    }

    getBoundingRect() {
        if (!this._rect) {
            const rect = new BoundingRect(Infinity, Infinity, -Infinity, -Infinity);
            for (let i = 0; i < this._displayables.length; i++) {
                const displayable = this._displayables[i];
                const childRect = displayable.getBoundingRect().clone();
                if (displayable.needLocalTransform()) {
                    childRect.applyTransform(displayable.getLocalTransform(m));
                }
                rect.union(childRect);
            }
            this._rect = rect;
        }
        return this._rect;
    }

    contain(x: number, y: number): boolean {
        const localPos = this.transformCoordToLocal(x, y);
        const rect = this.getBoundingRect();

        if (rect.contain(localPos[0], localPos[1])) {
            for (let i = 0; i < this._displayables.length; i++) {
                const displayable = this._displayables[i];
                if (displayable.contain(x, y)) {
                    return true;
                }
            }
        }
        return false;
    }

}