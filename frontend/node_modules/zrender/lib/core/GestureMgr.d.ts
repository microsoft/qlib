import { ZRRawTouchEvent } from './types';
import Displayable from '../graphic/Displayable';
export declare class GestureMgr {
    private _track;
    constructor();
    recognize(event: ZRRawTouchEvent, target: Displayable, root: HTMLElement): {
        type: string;
        target: Displayable<import("../graphic/Displayable").DisplayableProps>;
        event: ZRRawTouchEvent;
    };
    clear(): this;
    _doTrack(event: ZRRawTouchEvent, target: Displayable, root: HTMLElement): void;
    _recognize(event: ZRRawTouchEvent): {
        type: string;
        target: Displayable<import("../graphic/Displayable").DisplayableProps>;
        event: ZRRawTouchEvent;
    };
}
