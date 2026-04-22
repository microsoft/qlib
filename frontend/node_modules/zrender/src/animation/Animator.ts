/**
 * @module echarts/animation/Animator
 */

import Clip from './Clip';
import * as color from '../tool/color';
import {
    eqNaN,
    extend,
    isArrayLike,
    isFunction,
    isGradientObject,
    isNumber,
    isString,
    keys,
    logError,
    map
} from '../core/util';
import {ArrayLike, Dictionary} from '../core/types';
import easingFuncs, { AnimationEasing } from './easing';
import Animation from './Animation';
import { createCubicEasingFunc } from './cubicEasing';
import { isLinearGradient, isRadialGradient } from '../svg/helper';

type NumberArray = ArrayLike<number>
type InterpolatableType = string | number | NumberArray | NumberArray[];

interface ParsedColorStop {
    color: number[],
    offset: number
};

interface ParsedGradientObject {
    colorStops: ParsedColorStop[]
    x: number
    y: number
    global: boolean
}
interface ParsedLinearGradientObject extends ParsedGradientObject {
    x2: number
    y2: number
}
interface ParsedRadialGradientObject extends ParsedGradientObject {
    r: number
}

const arraySlice = Array.prototype.slice;

function interpolateNumber(p0: number, p1: number, percent: number): number {
    return (p1 - p0) * percent + p0;
}
function interpolate1DArray(
    out: NumberArray,
    p0: NumberArray,
    p1: NumberArray,
    percent: number
) {
    // TODO Handling different length TypedArray
    const len = p0.length;
    for (let i = 0; i < len; i++) {
        out[i] = interpolateNumber(p0[i], p1[i], percent);
    }
    return out;
}

function interpolate2DArray(
    out: NumberArray[],
    p0: NumberArray[],
    p1: NumberArray[],
    percent: number
) {
    const len = p0.length;
    // TODO differnt length on each item?
    const len2 = len && p0[0].length;
    for (let i = 0; i < len; i++) {
        if (!out[i]) {
            out[i] = [];
        }
        for (let j = 0; j < len2; j++) {
            out[i][j] = interpolateNumber(p0[i][j], p1[i][j], percent);
        }
    }
    return out;
}

function add1DArray(
    out: NumberArray,
    p0: NumberArray,
    p1: NumberArray,
    sign: 1 | -1
) {
    const len = p0.length;
    for (let i = 0; i < len; i++) {
        out[i] = p0[i] + p1[i] * sign;
    }
    return out;
}

function add2DArray(
    out: NumberArray[],
    p0: NumberArray[],
    p1: NumberArray[],
    sign: 1 | -1
) {
    const len = p0.length;
    const len2 = len && p0[0].length;
    for (let i = 0; i < len; i++) {
        if (!out[i]) {
            out[i] = [];
        }
        for (let j = 0; j < len2; j++) {
            out[i][j] = p0[i][j] + p1[i][j] * sign;
        }
    }
    return out;
}

function fillColorStops(val0: ParsedColorStop[], val1: ParsedColorStop[]) {
    const len0 = val0.length;
    const len1 = val1.length;

    const shorterArr = len0 > len1 ? val1 : val0;
    const shorterLen = Math.min(len0, len1);
    const last = shorterArr[shorterLen - 1] || { color: [0, 0, 0, 0], offset: 0 };
    for (let i = shorterLen; i < Math.max(len0, len1); i++) {
        // Use last color stop to fill the shorter array
        shorterArr.push({
            offset: last.offset,
            color: last.color.slice()
        });
    }
}
// arr0 is source array, arr1 is target array.
// Do some preprocess to avoid error happened when interpolating from arr0 to arr1
function fillArray(
    val0: NumberArray | NumberArray[],
    val1: NumberArray | NumberArray[],
    arrDim: 1 | 2
) {
    // TODO Handling different length TypedArray
    let arr0 = val0 as (number | number[])[];
    let arr1 = val1 as (number | number[])[];
    if (!arr0.push || !arr1.push) {
        return;
    }
    const arr0Len = arr0.length;
    const arr1Len = arr1.length;
    if (arr0Len !== arr1Len) {
        // FIXME Not work for TypedArray
        const isPreviousLarger = arr0Len > arr1Len;
        if (isPreviousLarger) {
            // Cut the previous
            arr0.length = arr1Len;
        }
        else {
            // Fill the previous
            for (let i = arr0Len; i < arr1Len; i++) {
                arr0.push(arrDim === 1 ? arr1[i] : arraySlice.call(arr1[i]));
            }
        }
    }
    // Handling NaN value
    const len2 = arr0[0] && (arr0[0] as number[]).length;
    for (let i = 0; i < arr0.length; i++) {
        if (arrDim === 1) {
            if (isNaN(arr0[i] as number)) {
                arr0[i] = arr1[i];
            }
        }
        else {
            for (let j = 0; j < len2; j++) {
                if (isNaN((arr0 as number[][])[i][j])) {
                    (arr0 as number[][])[i][j] = (arr1 as number[][])[i][j];
                }
            }
        }
    }
}

export function cloneValue(value: InterpolatableType) {
    if (isArrayLike(value)) {
        const len = value.length;
        if (isArrayLike(value[0])) {
            const ret = [];
            for (let i = 0; i < len; i++) {
                ret.push(arraySlice.call(value[i]));
            }
            return ret;
        }

        return arraySlice.call(value);
    }

    return value;
}

function rgba2String(rgba: number[]): string {
    rgba[0] = Math.floor(rgba[0]) || 0;
    rgba[1] = Math.floor(rgba[1]) || 0;
    rgba[2] = Math.floor(rgba[2]) || 0;
    rgba[3] = rgba[3] == null ? 1 : rgba[3];

    return 'rgba(' + rgba.join(',') + ')';
}

function guessArrayDim(value: ArrayLike<unknown>): 1 | 2 {
    return isArrayLike(value && (value as ArrayLike<unknown>)[0]) ? 2 : 1;
}

const VALUE_TYPE_NUMBER = 0;
const VALUE_TYPE_1D_ARRAY = 1;
const VALUE_TYPE_2D_ARRAY = 2;
const VALUE_TYPE_COLOR = 3;
const VALUE_TYPE_LINEAR_GRADIENT = 4;
const VALUE_TYPE_RADIAL_GRADIENT = 5;
// Other value type that can only use discrete animation.
const VALUE_TYPE_UNKOWN = 6;

type ValueType = 0 | 1 | 2 | 3 | 4 | 5 | 6;

type Keyframe = {
    time: number
    value: unknown
    percent: number
    // Raw value for discrete animation.
    rawValue: unknown

    easing?: AnimationEasing    // Raw easing
    easingFunc?: (percent: number) => number
    additiveValue?: unknown
}


function isGradientValueType(valType: ValueType): valType is 4 | 5 {
    return valType === VALUE_TYPE_LINEAR_GRADIENT || valType === VALUE_TYPE_RADIAL_GRADIENT;
}
function isArrayValueType(valType: ValueType): valType is 1 | 2 {
    return valType === VALUE_TYPE_1D_ARRAY || valType === VALUE_TYPE_2D_ARRAY;
}


let tmpRgba: number[] = [0, 0, 0, 0];

class Track {

    keyframes: Keyframe[] = []

    propName: string

    valType: ValueType

    discrete: boolean = false

    _invalid: boolean = false;

    private _finished: boolean

    private _needsSort: boolean = false

    private _additiveTrack: Track
    // Temporal storage for interpolated additive value.
    private _additiveValue: unknown

    // Info for run
    /**
     * Last frame
     */
    private _lastFr = 0
    /**
     * Percent of last frame.
     */
    private _lastFrP = 0

    constructor(propName: string) {
        this.propName = propName;
    }

    isFinished() {
        return this._finished;
    }

    setFinished() {
        this._finished = true;
        // Also set additive track to finished.
        // Make sure the final value stopped on the latest track
        if (this._additiveTrack) {
            this._additiveTrack.setFinished();
        }
    }

    needsAnimate() {
        return this.keyframes.length >= 1;
    }

    getAdditiveTrack() {
        return this._additiveTrack;
    }

    addKeyframe(time: number, rawValue: unknown, easing?: AnimationEasing) {
        this._needsSort = true;

        let keyframes = this.keyframes;
        let len = keyframes.length;

        let discrete = false;
        let valType: ValueType = VALUE_TYPE_UNKOWN;
        let value = rawValue;

        // Handling values only if it's possible to be interpolated.
        if (isArrayLike(rawValue)) {
            let arrayDim = guessArrayDim(rawValue);
            valType = arrayDim;
            // Not a number array.
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
                if (!isNaN(+rawValue)) {    // Can be string number like '2'
                    valType = VALUE_TYPE_NUMBER;
                }
                else {
                    const colorArray = color.parse(rawValue);
                    if (colorArray) {
                        value = colorArray;
                        valType = VALUE_TYPE_COLOR;
                    }
                }
            }
            else if (isGradientObject(rawValue)) {
                // TODO Color to gradient or gradient to color.
                const parsedGradient = extend({}, value) as unknown as ParsedGradientObject;
                parsedGradient.colorStops = map(rawValue.colorStops, colorStop => ({
                    offset: colorStop.offset,
                    color: color.parse(colorStop.color)
                }));
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
            // Inference type from the first keyframe.
            this.valType = valType;
        }
         // Not same value type or can't be interpolated.
        else if (valType !== this.valType || valType === VALUE_TYPE_UNKOWN) {
            discrete = true;
        }

        this.discrete = this.discrete || discrete;

        const kf: Keyframe = {
            time,
            value,
            rawValue,
            percent: 0
        };
        if (easing) {
            // Save the raw easing name to be used in css animation output
            kf.easing = easing;
            kf.easingFunc = isFunction(easing)
                ? easing
                : easingFuncs[easing] || createCubicEasingFunc(easing);
        }
        // Not check if value equal here.
        keyframes.push(kf);
        return kf;
    }

    prepare(maxTime: number, additiveTrack?: Track) {
        let kfs = this.keyframes;
        if (this._needsSort) {
            // Sort keyframe as ascending
            kfs.sort(function (a: Keyframe, b: Keyframe) {
                return a.time - b.time;
            });
        }

        const valType = this.valType;
        const kfsLen = kfs.length;
        const lastKf = kfs[kfsLen - 1];
        const isDiscrete = this.discrete;
        const isArr = isArrayValueType(valType);
        const isGradient = isGradientValueType(valType);

        for (let i = 0; i < kfsLen; i++) {
            const kf = kfs[i];
            const value = kf.value;
            const lastValue = lastKf.value;
            kf.percent = kf.time / maxTime;
            if (!isDiscrete) {
                if (isArr && i !== kfsLen - 1) {
                    // Align array with target frame.
                    fillArray(value as NumberArray, lastValue as NumberArray, valType);
                }
                else if (isGradient) {
                    fillColorStops(
                        (value as ParsedLinearGradientObject).colorStops,
                        (lastValue as ParsedLinearGradientObject).colorStops
                    );
                }
            }
        }

        // Only apply additive animaiton on INTERPOLABLE SAME TYPE values.
        if (
            !isDiscrete
            // TODO support gradient
            && valType !== VALUE_TYPE_RADIAL_GRADIENT
            && additiveTrack
            // If two track both will be animated and have same value format.
            && this.needsAnimate()
            && additiveTrack.needsAnimate()
            && valType === additiveTrack.valType
            && !additiveTrack._finished
        ) {
            this._additiveTrack = additiveTrack;

            const startValue = kfs[0].value;
            // Calculate difference
            for (let i = 0; i < kfsLen; i++) {
                if (valType === VALUE_TYPE_NUMBER) {
                    kfs[i].additiveValue = kfs[i].value as number - (startValue as number);
                }
                else if (valType === VALUE_TYPE_COLOR) {
                    kfs[i].additiveValue =
                        add1DArray([], kfs[i].value as NumberArray, startValue as NumberArray, -1);
                }
                else if (isArrayValueType(valType)) {
                    kfs[i].additiveValue = valType === VALUE_TYPE_1D_ARRAY
                        ? add1DArray([], kfs[i].value as NumberArray, startValue as NumberArray, -1)
                        : add2DArray([], kfs[i].value as NumberArray[], startValue as NumberArray[], -1);
                }
            }
        }
    }

    step(target: any, percent: number) {
        if (this._finished) {   // Track may be set to finished.
            return;
        }

        if (this._additiveTrack && this._additiveTrack._finished) {
            // Remove additive track if it's finished.
            this._additiveTrack = null;
        }
        const isAdditive = this._additiveTrack != null;
        const valueKey = isAdditive ? 'additiveValue' : 'value';

        const valType = this.valType;
        const keyframes = this.keyframes;
        const kfsNum = keyframes.length;
        const propName = this.propName;
        const isValueColor = valType === VALUE_TYPE_COLOR;
        // Find the range keyframes
        // kf1-----kf2---------current--------kf3
        // find kf2 and kf3 and do interpolation
        let frameIdx;
        const lastFrame = this._lastFr;
        const mathMin = Math.min;
        let frame;
        let nextFrame;
        if (kfsNum === 1) {
            frame = nextFrame = keyframes[0];
        }
        else {
            // In the easing function like elasticOut, percent may less than 0
            if (percent < 0) {
                frameIdx = 0;
            }
            else if (percent < this._lastFrP) {
                // Start from next key
                // PENDING start from lastFrame ?
                const start = mathMin(lastFrame + 1, kfsNum - 1);
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

        // Defensive coding.
        if (!(frame && nextFrame)) {
            return;
        }

        this._lastFr = frameIdx;
        this._lastFrP = percent;

        const interval = (nextFrame.percent - frame.percent);
        let w = interval === 0 ? 1 : mathMin((percent - frame.percent) / interval, 1);

        // Apply different easing of each keyframe.
        // Use easing specified in target frame.
        if (nextFrame.easingFunc) {
            w = nextFrame.easingFunc(w);
        }

        // If value is arr
        let targetArr = isAdditive ? this._additiveValue
            : (isValueColor ? tmpRgba : target[propName]);

        if ((isArrayValueType(valType) || isValueColor) && !targetArr) {
            targetArr = this._additiveValue = [];
        }

        if (this.discrete) {
            // use raw value without parse in discrete animation.
            target[propName] = w < 1 ? frame.rawValue : nextFrame.rawValue;
        }
        else if (isArrayValueType(valType)) {
            valType === VALUE_TYPE_1D_ARRAY
                ? interpolate1DArray(
                    targetArr as NumberArray,
                    frame[valueKey] as NumberArray,
                    nextFrame[valueKey] as NumberArray,
                    w
                )
                : interpolate2DArray(
                    targetArr as NumberArray[],
                    frame[valueKey] as NumberArray[],
                    nextFrame[valueKey] as NumberArray[],
                    w
                );
        }
        else if (isGradientValueType(valType)) {
            const val = frame[valueKey] as ParsedGradientObject;
            const nextVal = nextFrame[valueKey] as ParsedGradientObject;
            const isLinearGradient = valType === VALUE_TYPE_LINEAR_GRADIENT;
            target[propName] = {
                type: isLinearGradient ? 'linear' : 'radial',
                x: interpolateNumber(val.x, nextVal.x, w),
                y: interpolateNumber(val.y, nextVal.y, w),
                // TODO performance
                colorStops: map(val.colorStops, (colorStop, idx) => {
                    const nextColorStop = nextVal.colorStops[idx];
                    return {
                        offset: interpolateNumber(colorStop.offset, nextColorStop.offset, w),
                        color: rgba2String(interpolate1DArray(
                            [] as number[], colorStop.color, nextColorStop.color, w
                        ) as number[])
                    };
                }),
                global: nextVal.global
            };
            if (isLinearGradient) {
                // Linear
                target[propName].x2 = interpolateNumber(
                    (val as ParsedLinearGradientObject).x2, (nextVal as ParsedLinearGradientObject).x2, w
                );
                target[propName].y2 = interpolateNumber(
                    (val as ParsedLinearGradientObject).y2, (nextVal as ParsedLinearGradientObject).y2, w
                );
            }
            else {
                // Radial
                target[propName].r = interpolateNumber(
                    (val as ParsedRadialGradientObject).r, (nextVal as ParsedRadialGradientObject).r, w
                );
            }
        }
        else if (isValueColor) {
            interpolate1DArray(
                targetArr,
                frame[valueKey] as NumberArray,
                nextFrame[valueKey] as NumberArray,
                w
            );
            if (!isAdditive) {  // Convert to string later:)
                target[propName] = rgba2String(targetArr);
            }
        }
        else {
            const value = interpolateNumber(frame[valueKey] as number, nextFrame[valueKey] as number, w);
            if (isAdditive) {
                this._additiveValue = value;
            }
            else {
                target[propName] = value;
            }
        }

        // Add additive to target
        if (isAdditive) {
            this._addToTarget(target);
        }
    }

    private _addToTarget(target: any) {
        const valType = this.valType;
        const propName = this.propName;
        const additiveValue = this._additiveValue;

        if (valType === VALUE_TYPE_NUMBER) {
            // Add a difference value based on the change of previous frame.
            target[propName] = target[propName] + additiveValue;
        }
        else if (valType === VALUE_TYPE_COLOR) {
            // TODO reduce unnecessary parse
            color.parse(target[propName], tmpRgba);
            add1DArray(tmpRgba, tmpRgba, additiveValue as NumberArray, 1);
            target[propName] = rgba2String(tmpRgba);
        }
        else if (valType === VALUE_TYPE_1D_ARRAY) {
            add1DArray(target[propName], target[propName], additiveValue as NumberArray, 1);
        }
        else if (valType === VALUE_TYPE_2D_ARRAY) {
            add2DArray(target[propName], target[propName], additiveValue as NumberArray[], 1);
        }
    }
}


type DoneCallback = () => void;
type AbortCallback = () => void;
export type OnframeCallback<T> = (target: T, percent: number) => void;

export type AnimationPropGetter<T> = (target: T, key: string) => InterpolatableType;
export type AnimationPropSetter<T> = (target: T, key: string, value: InterpolatableType) => void;

export default class Animator<T> {

    animation?: Animation

    targetName?: string

    scope?: string

    __fromStateTransition?: string

    private _tracks: Dictionary<Track> = {}
    private _trackKeys: string[] = []

    private _target: T

    private _loop: boolean
    private _delay: number
    private _maxTime = 0

    /**
     * If force run regardless of empty tracks when duration is set.
     */
    private _force: boolean;

    /**
     * If animator is paused
     * @default false
     */
    private _paused: boolean
    // 0: Not started
    // 1: Invoked started
    // 2: Has been run for at least one frame.
    private _started = 0

    /**
     * If allow discrete animation
     * @default false
     */
    private _allowDiscrete: boolean

    private _additiveAnimators: Animator<any>[]

    private _doneCbs: DoneCallback[]
    private _onframeCbs: OnframeCallback<T>[]

    private _abortedCbs: AbortCallback[]

    private _clip: Clip = null

    constructor(
        target: T,
        loop: boolean,
        allowDiscreteAnimation?: boolean,  // If doing discrete animation on the values can't be interpolated
        additiveTo?: Animator<any>[]
    ) {
        this._target = target;
        this._loop = loop;
        if (loop && additiveTo) {
            logError('Can\' use additive animation on looped animation.');
            return;
        }
        this._additiveAnimators = additiveTo;

        this._allowDiscrete = allowDiscreteAnimation;
    }

    getMaxTime() {
        return this._maxTime;
    }

    getDelay() {
        return this._delay;
    }

    getLoop() {
        return this._loop;
    }

    getTarget() {
        return this._target;
    }

    /**
     * Target can be changed during animation
     * For example if style is changed during state change.
     * We need to change target to the new style object.
     */
    changeTarget(target: T) {
        this._target = target;
    }

    /**
     * Set Animation keyframe
     * @param time time of keyframe in ms
     * @param props key-value props of keyframe.
     * @param easing
     */
    when(time: number, props: Dictionary<any>, easing?: AnimationEasing) {
        return this.whenWithKeys(time, props, keys(props) as string[], easing);
    }


    // Fast path for add keyframes of aniamteTo
    whenWithKeys(time: number, props: Dictionary<any>, propNames: string[], easing?: AnimationEasing) {
        const tracks = this._tracks;
        for (let i = 0; i < propNames.length; i++) {
            const propName = propNames[i];

            let track = tracks[propName];
            if (!track) {
                track = tracks[propName] = new Track(propName);

                let initialValue;
                const additiveTrack = this._getAdditiveTrack(propName);
                if (additiveTrack) {
                    const addtiveTrackKfs = additiveTrack.keyframes;
                    const lastFinalKf = addtiveTrackKfs[addtiveTrackKfs.length - 1];
                    // Use the last state of additived animator.
                    initialValue = lastFinalKf && lastFinalKf.value;
                    if (additiveTrack.valType === VALUE_TYPE_COLOR && initialValue) {
                        // Convert to rgba string
                        initialValue = rgba2String(initialValue as number[]);
                    }
                }
                else {
                    initialValue = (this._target as any)[propName];
                }
                // Invalid value
                if (initialValue == null) {
                    // zrLog('Invalid property ' + propName);
                    continue;
                }
                // If time is <= 0
                //  Then props is given initialize value
                //  Note: initial percent can be negative, which means the initial value is before the animation start.
                // Else
                //  Initialize value from current prop value
                if (time > 0) {
                    track.addKeyframe(0, cloneValue(initialValue), easing);
                }

                this._trackKeys.push(propName);
            }
            track.addKeyframe(time, cloneValue(props[propName]), easing);
        }
        this._maxTime = Math.max(this._maxTime, time);
        return this;
    }

    pause() {
        this._clip.pause();
        this._paused = true;
    }

    resume() {
        this._clip.resume();
        this._paused = false;
    }

    isPaused(): boolean {
        return !!this._paused;
    }

    /**
     * Set duration of animator.
     * Will run this duration regardless the track max time or if trackes exits.
     * @param duration
     * @returns
     */
    duration(duration: number) {
        this._maxTime = duration;
        this._force = true;
        return this;
    }

    private _doneCallback() {
        this._setTracksFinished();
        // Clear clip
        this._clip = null;

        const doneList = this._doneCbs;
        if (doneList) {
            const len = doneList.length;
            for (let i = 0; i < len; i++) {
                doneList[i].call(this);
            }
        }
    }
    private _abortedCallback() {
        this._setTracksFinished();

        const animation = this.animation;
        const abortedList = this._abortedCbs;

        if (animation) {
            animation.removeClip(this._clip);
        }
        this._clip = null;

        if (abortedList) {
            for (let i = 0; i < abortedList.length; i++) {
                abortedList[i].call(this);
            }
        }
    }
    private _setTracksFinished() {
        const tracks = this._tracks;
        const tracksKeys = this._trackKeys;
        for (let i = 0; i < tracksKeys.length; i++) {
            tracks[tracksKeys[i]].setFinished();
        }
    }

    private _getAdditiveTrack(trackName: string): Track {
        let additiveTrack;
        const additiveAnimators = this._additiveAnimators;
        if (additiveAnimators) {
            for (let i = 0; i < additiveAnimators.length; i++) {
                const track = additiveAnimators[i].getTrack(trackName);
                if (track) {
                    // Use the track of latest animator.
                    additiveTrack = track;
                }
            }
        }
        return additiveTrack;
    }

    /**
     * Start the animation
     * @param easing
     * @return
     */
    start(easing?: AnimationEasing) {
        if (this._started > 0) {
            return;
        }
        this._started = 1;

        const self = this;

        const tracks: Track[] = [];
        const maxTime = this._maxTime || 0;
        for (let i = 0; i < this._trackKeys.length; i++) {
            const propName = this._trackKeys[i];
            const track = this._tracks[propName];
            const additiveTrack = this._getAdditiveTrack(propName);
            const kfs = track.keyframes;
            const kfsNum = kfs.length;
            track.prepare(maxTime, additiveTrack);
            if (track.needsAnimate()) {
                // Set value directly if discrete animation is not allowed.
                if (!this._allowDiscrete && track.discrete) {
                    const lastKf = kfs[kfsNum - 1];
                    // Set final value.
                    if (lastKf) {
                        // use raw value without parse.
                        (self._target as any)[track.propName] = lastKf.rawValue;
                    }
                    track.setFinished();
                }
                else {
                    tracks.push(track);
                }
            }
        }
        // Add during callback on the last clip
        if (tracks.length || this._force) {
            const clip = new Clip({
                life: maxTime,
                loop: this._loop,
                delay: this._delay || 0,
                onframe(percent: number) {
                    self._started = 2;
                    // Remove additived animator if it's finished.
                    // For the purpose of memory effeciency.
                    const additiveAnimators = self._additiveAnimators;
                    if (additiveAnimators) {
                        let stillHasAdditiveAnimator = false;
                        for (let i = 0; i < additiveAnimators.length; i++) {
                            if (additiveAnimators[i]._clip) {
                                stillHasAdditiveAnimator = true;
                                break;
                            }
                        }
                        if (!stillHasAdditiveAnimator) {
                            self._additiveAnimators = null;
                        }
                    }

                    for (let i = 0; i < tracks.length; i++) {
                        // NOTE: don't cache target outside.
                        // Because target may be changed.
                        tracks[i].step(self._target, percent);
                    }

                    const onframeList = self._onframeCbs;
                    if (onframeList) {
                        for (let i = 0; i < onframeList.length; i++) {
                            onframeList[i](self._target, percent);
                        }
                    }
                },
                ondestroy() {
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
            // This optimization will help the case that in the upper application
            // the view may be refreshed frequently, where animation will be
            // called repeatly but nothing changed.
            this._doneCallback();
        }

        return this;
    }
    /**
     * Stop animation
     * @param {boolean} forwardToLast If move to last frame before stop
     */
    stop(forwardToLast?: boolean) {
        if (!this._clip) {
            return;
        }
        const clip = this._clip;
        if (forwardToLast) {
            // Move to last frame before stop
            clip.onframe(1);
        }

        this._abortedCallback();
    }
    /**
     * Set when animation delay starts
     * @param time 单位ms
     */
    delay(time: number) {
        this._delay = time;
        return this;
    }
    /**
     * 添加动画每一帧的回调函数
     * @param callback
     */
    during(cb: OnframeCallback<T>) {
        if (cb) {
            if (!this._onframeCbs) {
                this._onframeCbs = [];
            }
            this._onframeCbs.push(cb);
        }
        return this;
    }
    /**
     * Add callback for animation end
     * @param cb
     */
    done(cb: DoneCallback) {
        if (cb) {
            if (!this._doneCbs) {
                this._doneCbs = [];
            }
            this._doneCbs.push(cb);
        }
        return this;
    }

    aborted(cb: AbortCallback) {
        if (cb) {
            if (!this._abortedCbs) {
                this._abortedCbs = [];
            }
            this._abortedCbs.push(cb);
        }
        return this;
    }

    getClip() {
        return this._clip;
    }

    getTrack(propName: string) {
        return this._tracks[propName];
    }

    getTracks() {
        return map(this._trackKeys, key => this._tracks[key]);
    }

    /**
     * Return true if animator is not available anymore.
     */
    stopTracks(propNames: string[], forwardToLast?: boolean): boolean {
        if (!propNames.length || !this._clip) {
            return true;
        }
        const tracks = this._tracks;
        const tracksKeys = this._trackKeys;

        for (let i = 0; i < propNames.length; i++) {
            const track = tracks[propNames[i]];
            if (track && !track.isFinished()) {
                if (forwardToLast) {
                    track.step(this._target, 1);
                }
                // If the track has not been run for at least one frame.
                // The property may be stayed at the final state. when setToFinal is set true.
                // For example:
                // Animate x from 0 to 100, then animate to 150 immediately.
                // We want the x is translated from 0 to 150, not 100 to 150.
                else if (this._started === 1) {
                    track.step(this._target, 0);
                }
                // Set track to finished
                track.setFinished();
            }
        }
        let allAborted = true;
        for (let i = 0; i < tracksKeys.length; i++) {
            if (!tracks[tracksKeys[i]].isFinished()) {
                allAborted = false;
                break;
            }
        }
        // Remove clip if all tracks has been aborted.
        if (allAborted) {
            this._abortedCallback();
        }

        return allAborted;
    }

    /**
     * Save values of final state to target.
     * It is mainly used in state mangement. When state is switching during animation.
     * We need to save final state of animation to the normal state. Not interpolated value.
     *
     * @param target
     * @param trackKeys
     * @param firstOrLast If save first frame or last frame
     */
    saveTo(
        target: T,
        trackKeys?: readonly string[],
        firstOrLast?: boolean
    ) {
        if (!target) {  // DO nothing if target is not given.
            return;
        }

        trackKeys = trackKeys || this._trackKeys;

        for (let i = 0; i < trackKeys.length; i++) {
            const propName = trackKeys[i];
            const track = this._tracks[propName];
            if (!track || track.isFinished()) {   // Ignore finished track.
                continue;
            }
            const kfs = track.keyframes;
            const kf = kfs[firstOrLast ? 0 : kfs.length - 1];
            if (kf) {
                // TODO CLONE?
                // Use raw value without parse.
                (target as any)[propName] = cloneValue(kf.rawValue as any);
            }
        }
    }

    // Change final value after animator has been started.
    // NOTE: Be careful to use it.
    __changeFinalValue(finalProps: Dictionary<any>, trackKeys?: readonly string[]) {
        trackKeys = trackKeys || keys(finalProps);

        for (let i = 0; i < trackKeys.length; i++) {
            const propName = trackKeys[i];

            const track = this._tracks[propName];
            if (!track) {
                continue;
            }

            const kfs = track.keyframes;
            if (kfs.length > 1) {
                // Remove the original last kf and add again.
                const lastKf = kfs.pop();

                track.addKeyframe(lastKf.time, finalProps[propName]);
                // Prepare again.
                track.prepare(this._maxTime, track.getAdditiveTrack());
            }
        }
    }
}

export type AnimatorTrack = Track;