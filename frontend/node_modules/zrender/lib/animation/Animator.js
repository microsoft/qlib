import Clip from './Clip.js';
import * as color from '../tool/color.js';
import { eqNaN, extend, isArrayLike, isFunction, isGradientObject, isNumber, isString, keys, logError, map } from '../core/util.js';
import easingFuncs from './easing.js';
import { createCubicEasingFunc } from './cubicEasing.js';
import { isLinearGradient, isRadialGradient } from '../svg/helper.js';
;
var arraySlice = Array.prototype.slice;
function interpolateNumber(p0, p1, percent) {
    return (p1 - p0) * percent + p0;
}
function interpolate1DArray(out, p0, p1, percent) {
    var len = p0.length;
    for (var i = 0; i < len; i++) {
        out[i] = interpolateNumber(p0[i], p1[i], percent);
    }
    return out;
}
function interpolate2DArray(out, p0, p1, percent) {
    var len = p0.length;
    var len2 = len && p0[0].length;
    for (var i = 0; i < len; i++) {
        if (!out[i]) {
            out[i] = [];
        }
        for (var j = 0; j < len2; j++) {
            out[i][j] = interpolateNumber(p0[i][j], p1[i][j], percent);
        }
    }
    return out;
}
function add1DArray(out, p0, p1, sign) {
    var len = p0.length;
    for (var i = 0; i < len; i++) {
        out[i] = p0[i] + p1[i] * sign;
    }
    return out;
}
function add2DArray(out, p0, p1, sign) {
    var len = p0.length;
    var len2 = len && p0[0].length;
    for (var i = 0; i < len; i++) {
        if (!out[i]) {
            out[i] = [];
        }
        for (var j = 0; j < len2; j++) {
            out[i][j] = p0[i][j] + p1[i][j] * sign;
        }
    }
    return out;
}
function fillColorStops(val0, val1) {
    var len0 = val0.length;
    var len1 = val1.length;
    var shorterArr = len0 > len1 ? val1 : val0;
    var shorterLen = Math.min(len0, len1);
    var last = shorterArr[shorterLen - 1] || { color: [0, 0, 0, 0], offset: 0 };
    for (var i = shorterLen; i < Math.max(len0, len1); i++) {
        shorterArr.push({
            offset: last.offset,
            color: last.color.slice()
        });
    }
}
function fillArray(val0, val1, arrDim) {
    var arr0 = val0;
    var arr1 = val1;
    if (!arr0.push || !arr1.push) {
        return;
    }
    var arr0Len = arr0.length;
    var arr1Len = arr1.length;
    if (arr0Len !== arr1Len) {
        var isPreviousLarger = arr0Len > arr1Len;
        if (isPreviousLarger) {
            arr0.length = arr1Len;
        }
        else {
            for (var i = arr0Len; i < arr1Len; i++) {
                arr0.push(arrDim === 1 ? arr1[i] : arraySlice.call(arr1[i]));
            }
        }
    }
    var len2 = arr0[0] && arr0[0].length;
    for (var i = 0; i < arr0.length; i++) {
        if (arrDim === 1) {
            if (isNaN(arr0[i])) {
                arr0[i] = arr1[i];
            }
        }
        else {
            for (var j = 0; j < len2; j++) {
                if (isNaN(arr0[i][j])) {
                    arr0[i][j] = arr1[i][j];
                }
            }
        }
    }
}
export function cloneValue(value) {
    if (isArrayLike(value)) {
        var len = value.length;
        if (isArrayLike(value[0])) {
            var ret = [];
            for (var i = 0; i < len; i++) {
                ret.push(arraySlice.call(value[i]));
            }
            return ret;
        }
        return arraySlice.call(value);
    }
    return value;
}
function rgba2String(rgba) {
    rgba[0] = Math.floor(rgba[0]) || 0;
    rgba[1] = Math.floor(rgba[1]) || 0;
    rgba[2] = Math.floor(rgba[2]) || 0;
    rgba[3] = rgba[3] == null ? 1 : rgba[3];
    return 'rgba(' + rgba.join(',') + ')';
}
function guessArrayDim(value) {
    return isArrayLike(value && value[0]) ? 2 : 1;
}
var VALUE_TYPE_NUMBER = 0;
var VALUE_TYPE_1D_ARRAY = 1;
var VALUE_TYPE_2D_ARRAY = 2;
var VALUE_TYPE_COLOR = 3;
var VALUE_TYPE_LINEAR_GRADIENT = 4;
var VALUE_TYPE_RADIAL_GRADIENT = 5;
var VALUE_TYPE_UNKOWN = 6;
function isGradientValueType(valType) {
    return valType === VALUE_TYPE_LINEAR_GRADIENT || valType === VALUE_TYPE_RADIAL_GRADIENT;
}
function isArrayValueType(valType) {
    return valType === VALUE_TYPE_1D_ARRAY || valType === VALUE_TYPE_2D_ARRAY;
}
var tmpRgba = [0, 0, 0, 0];
var Track = (function () {
    function Track(propName) {
        this.keyframes = [];
        this.discrete = false;
        this._invalid = false;
        this._needsSort = false;
        this._lastFr = 0;
        this._lastFrP = 0;
        this.propName = propName;
    }
    Track.prototype.isFinished = function () {
        return this._finished;
    };
    Track.prototype.setFinished = function () {
        this._finished = true;
        if (this._additiveTrack) {
            this._additiveTrack.setFinished();
        }
    };
    Track.prototype.needsAnimate = function () {
        return this.keyframes.length >= 1;
    };
    Track.prototype.getAdditiveTrack = function () {
        return this._additiveTrack;
    };
    Track.prototype.addKeyframe = function (time, rawValue, easing) {
        this._needsSort = true;
        var keyframes = this.keyframes;
        var len = keyframes.length;
        var discrete = false;
        var valType = VALUE_TYPE_UNKOWN;
        var value = rawValue;
        if (isArrayLike(rawValue)) {
            var arrayDim = guessArrayDim(rawValue);
            valType = arrayDim;
            if (arrayDim === 1 && !isNumber(rawValue[0])
                || arrayDim === 2 && !isNumber(rawValue[0][0])) {
                discrete = true;
            }
        }
        else {
            if (isNumber(rawValue) && !eqNaN(rawValue)) {
                valType = VALUE_TYPE_NUMBER;
            }
            else if (isString(rawValue)) {
                if (!isNaN(+rawValue)) {
                    valType = VALUE_TYPE_NUMBER;
                }
                else {
                    var colorArray = color.parse(rawValue);
                    if (colorArray) {
                        value = colorArray;
                        valType = VALUE_TYPE_COLOR;
                    }
                }
            }
            else if (isGradientObject(rawValue)) {
                var parsedGradient = extend({}, value);
                parsedGradient.colorStops = map(rawValue.colorStops, function (colorStop) { return ({
                    offset: colorStop.offset,
                    color: color.parse(colorStop.color)
                }); });
                if (isLinearGradient(rawValue)) {
                    valType = VALUE_TYPE_LINEAR_GRADIENT;
                }
                else if (isRadialGradient(rawValue)) {
                    valType = VALUE_TYPE_RADIAL_GRADIENT;
                }
                value = parsedGradient;
            }
        }
        if (len === 0) {
            this.valType = valType;
        }
        else if (valType !== this.valType || valType === VALUE_TYPE_UNKOWN) {
            discrete = true;
        }
        this.discrete = this.discrete || discrete;
        var kf = {
            time: time,
            value: value,
            rawValue: rawValue,
            percent: 0
        };
        if (easing) {
            kf.easing = easing;
            kf.easingFunc = isFunction(easing)
                ? easing
                : easingFuncs[easing] || createCubicEasingFunc(easing);
        }
        keyframes.push(kf);
        return kf;
    };
    Track.prototype.prepare = function (maxTime, additiveTrack) {
        var kfs = this.keyframes;
        if (this._needsSort) {
            kfs.sort(function (a, b) {
                return a.time - b.time;
            });
        }
        var valType = this.valType;
        var kfsLen = kfs.length;
        var lastKf = kfs[kfsLen - 1];
        var isDiscrete = this.discrete;
        var isArr = isArrayValueType(valType);
        var isGradient = isGradientValueType(valType);
        for (var i = 0; i < kfsLen; i++) {
            var kf = kfs[i];
            var value = kf.value;
            var lastValue = lastKf.value;
            kf.percent = kf.time / maxTime;
            if (!isDiscrete) {
                if (isArr && i !== kfsLen - 1) {
                    fillArray(value, lastValue, valType);
                }
                else if (isGradient) {
                    fillColorStops(value.colorStops, lastValue.colorStops);
                }
            }
        }
        if (!isDiscrete
            && valType !== VALUE_TYPE_RADIAL_GRADIENT
            && additiveTrack
            && this.needsAnimate()
            && additiveTrack.needsAnimate()
            && valType === additiveTrack.valType
            && !additiveTrack._finished) {
            this._additiveTrack = additiveTrack;
            var startValue = kfs[0].value;
            for (var i = 0; i < kfsLen; i++) {
                if (valType === VALUE_TYPE_NUMBER) {
                    kfs[i].additiveValue = kfs[i].value - startValue;
                }
                else if (valType === VALUE_TYPE_COLOR) {
                    kfs[i].additiveValue =
                        add1DArray([], kfs[i].value, startValue, -1);
                }
                else if (isArrayValueType(valType)) {
                    kfs[i].additiveValue = valType === VALUE_TYPE_1D_ARRAY
                        ? add1DArray([], kfs[i].value, startValue, -1)
                        : add2DArray([], kfs[i].value, startValue, -1);
                }
            }
        }
    };
    Track.prototype.step = function (target, percent) {
        if (this._finished) {
            return;
        }
        if (this._additiveTrack && this._additiveTrack._finished) {
            this._additiveTrack = null;
        }
        var isAdditive = this._additiveTrack != null;
        var valueKey = isAdditive ? 'additiveValue' : 'value';
        var valType = this.valType;
        var keyframes = this.keyframes;
        var kfsNum = keyframes.length;
        var propName = this.propName;
        var isValueColor = valType === VALUE_TYPE_COLOR;
        var frameIdx;
        var lastFrame = this._lastFr;
        var mathMin = Math.min;
        var frame;
        var nextFrame;
        if (kfsNum === 1) {
            frame = nextFrame = keyframes[0];
        }
        else {
            if (percent < 0) {
                frameIdx = 0;
            }
            else if (percent < this._lastFrP) {
                var start = mathMin(lastFrame + 1, kfsNum - 1);
                for (frameIdx = start; frameIdx >= 0; frameIdx--) {
                    if (keyframes[frameIdx].percent <= percent) {
                        break;
                    }
                }
                frameIdx = mathMin(frameIdx, kfsNum - 2);
            }
            else {
                for (frameIdx = lastFrame; frameIdx < kfsNum; frameIdx++) {
                    if (keyframes[frameIdx].percent > percent) {
                        break;
                    }
                }
                frameIdx = mathMin(frameIdx - 1, kfsNum - 2);
            }
            nextFrame = keyframes[frameIdx + 1];
            frame = keyframes[frameIdx];
        }
        if (!(frame && nextFrame)) {
            return;
        }
        this._lastFr = frameIdx;
        this._lastFrP = percent;
        var interval = (nextFrame.percent - frame.percent);
        var w = interval === 0 ? 1 : mathMin((percent - frame.percent) / interval, 1);
        if (nextFrame.easingFunc) {
            w = nextFrame.easingFunc(w);
        }
        var targetArr = isAdditive ? this._additiveValue
            : (isValueColor ? tmpRgba : target[propName]);
        if ((isArrayValueType(valType) || isValueColor) && !targetArr) {
            targetArr = this._additiveValue = [];
        }
        if (this.discrete) {
            target[propName] = w < 1 ? frame.rawValue : nextFrame.rawValue;
        }
        else if (isArrayValueType(valType)) {
            valType === VALUE_TYPE_1D_ARRAY
                ? interpolate1DArray(targetArr, frame[valueKey], nextFrame[valueKey], w)
                : interpolate2DArray(targetArr, frame[valueKey], nextFrame[valueKey], w);
        }
        else if (isGradientValueType(valType)) {
            var val = frame[valueKey];
            var nextVal_1 = nextFrame[valueKey];
            var isLinearGradient_1 = valType === VALUE_TYPE_LINEAR_GRADIENT;
            target[propName] = {
                type: isLinearGradient_1 ? 'linear' : 'radial',
                x: interpolateNumber(val.x, nextVal_1.x, w),
                y: interpolateNumber(val.y, nextVal_1.y, w),
                colorStops: map(val.colorStops, function (colorStop, idx) {
                    var nextColorStop = nextVal_1.colorStops[idx];
                    return {
                        offset: interpolateNumber(colorStop.offset, nextColorStop.offset, w),
                        color: rgba2String(interpolate1DArray([], colorStop.color, nextColorStop.color, w))
                    };
                }),
                global: nextVal_1.global
            };
            if (isLinearGradient_1) {
                target[propName].x2 = interpolateNumber(val.x2, nextVal_1.x2, w);
                target[propName].y2 = interpolateNumber(val.y2, nextVal_1.y2, w);
            }
            else {
                target[propName].r = interpolateNumber(val.r, nextVal_1.r, w);
            }
        }
        else if (isValueColor) {
            interpolate1DArray(targetArr, frame[valueKey], nextFrame[valueKey], w);
            if (!isAdditive) {
                target[propName] = rgba2String(targetArr);
            }
        }
        else {
            var value = interpolateNumber(frame[valueKey], nextFrame[valueKey], w);
            if (isAdditive) {
                this._additiveValue = value;
            }
            else {
                target[propName] = value;
            }
        }
        if (isAdditive) {
            this._addToTarget(target);
        }
    };
    Track.prototype._addToTarget = function (target) {
        var valType = this.valType;
        var propName = this.propName;
        var additiveValue = this._additiveValue;
        if (valType === VALUE_TYPE_NUMBER) {
            target[propName] = target[propName] + additiveValue;
        }
        else if (valType === VALUE_TYPE_COLOR) {
            color.parse(target[propName], tmpRgba);
            add1DArray(tmpRgba, tmpRgba, additiveValue, 1);
            target[propName] = rgba2String(tmpRgba);
        }
        else if (valType === VALUE_TYPE_1D_ARRAY) {
            add1DArray(target[propName], target[propName], additiveValue, 1);
        }
        else if (valType === VALUE_TYPE_2D_ARRAY) {
            add2DArray(target[propName], target[propName], additiveValue, 1);
        }
    };
    return Track;
}());
var Animator = (function () {
    function Animator(target, loop, allowDiscreteAnimation, additiveTo) {
        this._tracks = {};
        this._trackKeys = [];
        this._maxTime = 0;
        this._started = 0;
        this._clip = null;
        this._target = target;
        this._loop = loop;
        if (loop && additiveTo) {
            logError('Can\' use additive animation on looped animation.');
            return;
        }
        this._additiveAnimators = additiveTo;
        this._allowDiscrete = allowDiscreteAnimation;
    }
    Animator.prototype.getMaxTime = function () {
        return this._maxTime;
    };
    Animator.prototype.getDelay = function () {
        return this._delay;
    };
    Animator.prototype.getLoop = function () {
        return this._loop;
    };
    Animator.prototype.getTarget = function () {
        return this._target;
    };
    Animator.prototype.changeTarget = function (target) {
        this._target = target;
    };
    Animator.prototype.when = function (time, props, easing) {
        return this.whenWithKeys(time, props, keys(props), easing);
    };
    Animator.prototype.whenWithKeys = function (time, props, propNames, easing) {
        var tracks = this._tracks;
        for (var i = 0; i < propNames.length; i++) {
            var propName = propNames[i];
            var track = tracks[propName];
            if (!track) {
                track = tracks[propName] = new Track(propName);
                var initialValue = void 0;
                var additiveTrack = this._getAdditiveTrack(propName);
                if (additiveTrack) {
                    var addtiveTrackKfs = additiveTrack.keyframes;
                    var lastFinalKf = addtiveTrackKfs[addtiveTrackKfs.length - 1];
                    initialValue = lastFinalKf && lastFinalKf.value;
                    if (additiveTrack.valType === VALUE_TYPE_COLOR && initialValue) {
                        initialValue = rgba2String(initialValue);
                    }
                }
                else {
                    initialValue = this._target[propName];
                }
                if (initialValue == null) {
                    continue;
                }
                if (time > 0) {
                    track.addKeyframe(0, cloneValue(initialValue), easing);
                }
                this._trackKeys.push(propName);
            }
            track.addKeyframe(time, cloneValue(props[propName]), easing);
        }
        this._maxTime = Math.max(this._maxTime, time);
        return this;
    };
    Animator.prototype.pause = function () {
        this._clip.pause();
        this._paused = true;
    };
    Animator.prototype.resume = function () {
        this._clip.resume();
        this._paused = false;
    };
    Animator.prototype.isPaused = function () {
        return !!this._paused;
    };
    Animator.prototype.duration = function (duration) {
        this._maxTime = duration;
        this._force = true;
        return this;
    };
    Animator.prototype._doneCallback = function () {
        this._setTracksFinished();
        this._clip = null;
        var doneList = this._doneCbs;
        if (doneList) {
            var len = doneList.length;
            for (var i = 0; i < len; i++) {
                doneList[i].call(this);
            }
        }
    };
    Animator.prototype._abortedCallback = function () {
        this._setTracksFinished();
        var animation = this.animation;
        var abortedList = this._abortedCbs;
        if (animation) {
            animation.removeClip(this._clip);
        }
        this._clip = null;
        if (abortedList) {
            for (var i = 0; i < abortedList.length; i++) {
                abortedList[i].call(this);
            }
        }
    };
    Animator.prototype._setTracksFinished = function () {
        var tracks = this._tracks;
        var tracksKeys = this._trackKeys;
        for (var i = 0; i < tracksKeys.length; i++) {
            tracks[tracksKeys[i]].setFinished();
        }
    };
    Animator.prototype._getAdditiveTrack = function (trackName) {
        var additiveTrack;
        var additiveAnimators = this._additiveAnimators;
        if (additiveAnimators) {
            for (var i = 0; i < additiveAnimators.length; i++) {
                var track = additiveAnimators[i].getTrack(trackName);
                if (track) {
                    additiveTrack = track;
                }
            }
        }
        return additiveTrack;
    };
    Animator.prototype.start = function (easing) {
        if (this._started > 0) {
            return;
        }
        this._started = 1;
        var self = this;
        var tracks = [];
        var maxTime = this._maxTime || 0;
        for (var i = 0; i < this._trackKeys.length; i++) {
            var propName = this._trackKeys[i];
            var track = this._tracks[propName];
            var additiveTrack = this._getAdditiveTrack(propName);
            var kfs = track.keyframes;
            var kfsNum = kfs.length;
            track.prepare(maxTime, additiveTrack);
            if (track.needsAnimate()) {
                if (!this._allowDiscrete && track.discrete) {
                    var lastKf = kfs[kfsNum - 1];
                    if (lastKf) {
                        self._target[track.propName] = lastKf.rawValue;
                    }
                    track.setFinished();
                }
                else {
                    tracks.push(track);
                }
            }
        }
        if (tracks.length || this._force) {
            var clip = new Clip({
                life: maxTime,
                loop: this._loop,
                delay: this._delay || 0,
                onframe: function (percent) {
                    self._started = 2;
                    var additiveAnimators = self._additiveAnimators;
                    if (additiveAnimators) {
                        var stillHasAdditiveAnimator = false;
                        for (var i = 0; i < additiveAnimators.length; i++) {
                            if (additiveAnimators[i]._clip) {
                                stillHasAdditiveAnimator = true;
                                break;
                            }
                        }
                        if (!stillHasAdditiveAnimator) {
                            self._additiveAnimators = null;
                        }
                    }
                    for (var i = 0; i < tracks.length; i++) {
                        tracks[i].step(self._target, percent);
                    }
                    var onframeList = self._onframeCbs;
                    if (onframeList) {
                        for (var i = 0; i < onframeList.length; i++) {
                            onframeList[i](self._target, percent);
                        }
                    }
                },
                ondestroy: function () {
                    self._doneCallback();
                }
            });
            this._clip = clip;
            if (this.animation) {
                this.animation.addClip(clip);
            }
            if (easing) {
                clip.setEasing(easing);
            }
        }
        else {
            this._doneCallback();
        }
        return this;
    };
    Animator.prototype.stop = function (forwardToLast) {
        if (!this._clip) {
            return;
        }
        var clip = this._clip;
        if (forwardToLast) {
            clip.onframe(1);
        }
        this._abortedCallback();
    };
    Animator.prototype.delay = function (time) {
        this._delay = time;
        return this;
    };
    Animator.prototype.during = function (cb) {
        if (cb) {
            if (!this._onframeCbs) {
                this._onframeCbs = [];
            }
            this._onframeCbs.push(cb);
        }
        return this;
    };
    Animator.prototype.done = function (cb) {
        if (cb) {
            if (!this._doneCbs) {
                this._doneCbs = [];
            }
            this._doneCbs.push(cb);
        }
        return this;
    };
    Animator.prototype.aborted = function (cb) {
        if (cb) {
            if (!this._abortedCbs) {
                this._abortedCbs = [];
            }
            this._abortedCbs.push(cb);
        }
        return this;
    };
    Animator.prototype.getClip = function () {
        return this._clip;
    };
    Animator.prototype.getTrack = function (propName) {
        return this._tracks[propName];
    };
    Animator.prototype.getTracks = function () {
        var _this = this;
        return map(this._trackKeys, function (key) { return _this._tracks[key]; });
    };
    Animator.prototype.stopTracks = function (propNames, forwardToLast) {
        if (!propNames.length || !this._clip) {
            return true;
        }
        var tracks = this._tracks;
        var tracksKeys = this._trackKeys;
        for (var i = 0; i < propNames.length; i++) {
            var track = tracks[propNames[i]];
            if (track && !track.isFinished()) {
                if (forwardToLast) {
                    track.step(this._target, 1);
                }
                else if (this._started === 1) {
                    track.step(this._target, 0);
                }
                track.setFinished();
            }
        }
        var allAborted = true;
        for (var i = 0; i < tracksKeys.length; i++) {
            if (!tracks[tracksKeys[i]].isFinished()) {
                allAborted = false;
                break;
            }
        }
        if (allAborted) {
            this._abortedCallback();
        }
        return allAborted;
    };
    Animator.prototype.saveTo = function (target, trackKeys, firstOrLast) {
        if (!target) {
            return;
        }
        trackKeys = trackKeys || this._trackKeys;
        for (var i = 0; i < trackKeys.length; i++) {
            var propName = trackKeys[i];
            var track = this._tracks[propName];
            if (!track || track.isFinished()) {
                continue;
            }
            var kfs = track.keyframes;
            var kf = kfs[firstOrLast ? 0 : kfs.length - 1];
            if (kf) {
                target[propName] = cloneValue(kf.rawValue);
            }
        }
    };
    Animator.prototype.__changeFinalValue = function (finalProps, trackKeys) {
        trackKeys = trackKeys || keys(finalProps);
        for (var i = 0; i < trackKeys.length; i++) {
            var propName = trackKeys[i];
            var track = this._tracks[propName];
            if (!track) {
                continue;
            }
            var kfs = track.keyframes;
            if (kfs.length > 1) {
                var lastKf = kfs.pop();
                track.addKeyframe(lastKf.time, finalProps[propName]);
                track.prepare(this._maxTime, track.getAdditiveTrack());
            }
        }
    };
    return Animator;
}());
export default Animator;
