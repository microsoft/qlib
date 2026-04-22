/**
 * Only implements needed gestures for mobile.
 */

import * as eventUtil from './event';
import { ZRRawTouchEvent, ZRPinchEvent, Dictionary } from './types';
import Displayable from '../graphic/Displayable';

interface TrackItem {
    points: number[][]
    touches: Touch[]
    target: Displayable,
    event: ZRRawTouchEvent
}

export class GestureMgr {

    private _track: TrackItem[] = []

    constructor() {}

    recognize(event: ZRRawTouchEvent, target: Displayable, root: HTMLElement) {
        this._doTrack(event, target, root);
        return this._recognize(event);
    }

    clear() {
        this._track.length = 0;
        return this;
    }

    _doTrack(event: ZRRawTouchEvent, target: Displayable, root: HTMLElement) {
        const touches = event.touches;

        if (!touches) {
            return;
        }

        const trackItem: TrackItem = {
            points: [],
            touches: [],
            target: target,
            event: event
        };

        for (let i = 0, len = touches.length; i < len; i++) {
            const touch = touches[i];
            const pos = eventUtil.clientToLocal(root, touch, {});
            trackItem.points.push([pos.zrX, pos.zrY]);
            trackItem.touches.push(touch);
        }

        this._track.push(trackItem);
    }

    _recognize(event: ZRRawTouchEvent) {
        for (let eventName in recognizers) {
            if (recognizers.hasOwnProperty(eventName)) {
                const gestureInfo = recognizers[eventName](this._track, event);
                if (gestureInfo) {
                    return gestureInfo;
                }
            }
        }
    }
}

function dist(pointPair: number[][]): number {
    const dx = pointPair[1][0] - pointPair[0][0];
    const dy = pointPair[1][1] - pointPair[0][1];

    return Math.sqrt(dx * dx + dy * dy);
}

function center(pointPair: number[][]): number[] {
    return [
        (pointPair[0][0] + pointPair[1][0]) / 2,
        (pointPair[0][1] + pointPair[1][1]) / 2
    ];
}

type Recognizer = (tracks: TrackItem[], event: ZRRawTouchEvent) => {
    type: string
    target: Displayable
    event: ZRRawTouchEvent
}

const recognizers: Dictionary<Recognizer> = {

    pinch: function (tracks: TrackItem[], event: ZRRawTouchEvent) {
        const trackLen = tracks.length;

        if (!trackLen) {
            return;
        }

        const pinchEnd = (tracks[trackLen - 1] || {}).points;
        const pinchPre = (tracks[trackLen - 2] || {}).points || pinchEnd;

        if (pinchPre
            && pinchPre.length > 1
            && pinchEnd
            && pinchEnd.length > 1
        ) {
            let pinchScale = dist(pinchEnd) / dist(pinchPre);
            !isFinite(pinchScale) && (pinchScale = 1);

            (event as ZRPinchEvent).pinchScale = pinchScale;

            const pinchCenter = center(pinchEnd);
            (event as ZRPinchEvent).pinchX = pinchCenter[0];
            (event as ZRPinchEvent).pinchY = pinchCenter[1];

            return {
                type: 'pinch',
                target: tracks[0].target,
                event: event
            };
        }
    }

    // Only pinch currently.
};