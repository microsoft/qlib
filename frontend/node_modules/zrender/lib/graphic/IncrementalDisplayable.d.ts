import Displayble from './Displayable';
import BoundingRect from '../core/BoundingRect';
export default class IncrementalDisplayable extends Displayble {
    notClear: boolean;
    incremental: boolean;
    private _displayables;
    private _temporaryDisplayables;
    private _cursor;
    traverse<T>(cb: (this: T, el: this) => void, context: T): void;
    useStyle(): void;
    getCursor(): number;
    innerAfterBrush(): void;
    clearDisplaybles(): void;
    clearTemporalDisplayables(): void;
    addDisplayable(displayable: Displayble, notPersistent?: boolean): void;
    addDisplayables(displayables: Displayble[], notPersistent?: boolean): void;
    getDisplayables(): Displayble[];
    getTemporalDisplayables(): Displayble[];
    eachPendingDisplayable(cb: (displayable: Displayble) => void): void;
    update(): void;
    getBoundingRect(): BoundingRect;
    contain(x: number, y: number): boolean;
}
