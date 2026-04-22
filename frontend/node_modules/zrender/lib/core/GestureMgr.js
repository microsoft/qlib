import * as eventUtil from './event.js';
var GestureMgr = (function () {
    function GestureMgr() {
        this._track = [];
    }
    GestureMgr.prototype.recognize = function (event, target, root) {
        this._doTrack(event, target, root);
        return this._recognize(event);
    };
    GestureMgr.prototype.clear = function () {
        this._track.length = 0;
        return this;
    };
    GestureMgr.prototype._doTrack = function (event, target, root) {
        var touches = event.touches;
        if (!touches) {
            return;
        }
        var trackItem = {
            points: [],
            touches: [],
            target: target,
            event: event
        };
        for (var i = 0, len = touches.length; i < len; i++) {
            var touch = touches[i];
            var pos = eventUtil.clientToLocal(root, touch, {});
            trackItem.points.push([pos.zrX, pos.zrY]);
            trackItem.touches.push(touch);
        }
        this._track.push(trackItem);
    };
    GestureMgr.prototype._recognize = function (event) {
        for (var eventName in recognizers) {
            if (recognizers.hasOwnProperty(eventName)) {
                var gestureInfo = recognizers[eventName](this._track, event);
                if (gestureInfo) {
                    return gestureInfo;
                }
            }
        }
    };
    return GestureMgr;
}());
export { GestureMgr };
function dist(pointPair) {
    var dx = pointPair[1][0] - pointPair[0][0];
    var dy = pointPair[1][1] - pointPair[0][1];
    return Math.sqrt(dx * dx + dy * dy);
}
function center(pointPair) {
    return [
        (pointPair[0][0] + pointPair[1][0]) / 2,
        (pointPair[0][1] + pointPair[1][1]) / 2
    ];
}
var recognizers = {
    pinch: function (tracks, event) {
        var trackLen = tracks.length;
        if (!trackLen) {
            return;
        }
        var pinchEnd = (tracks[trackLen - 1] || {}).points;
        var pinchPre = (tracks[trackLen - 2] || {}).points || pinchEnd;
        if (pinchPre
            && pinchPre.length > 1
            && pinchEnd
            && pinchEnd.length > 1) {
            var pinchScale = dist(pinchEnd) / dist(pinchPre);
            !isFinite(pinchScale) && (pinchScale = 1);
            event.pinchScale = pinchScale;
            var pinchCenter = center(pinchEnd);
            event.pinchX = pinchCenter[0];
            event.pinchY = pinchCenter[1];
            return {
                type: 'pinch',
                target: tracks[0].target,
                event: event
            };
        }
    }
};
