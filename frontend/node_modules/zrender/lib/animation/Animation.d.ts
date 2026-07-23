import Eventful from '../core/Eventful';
import Animator from './Animator';
import Clip from './Clip';
export declare function getTime(): number;
interface Stage {
    update?: () => void;
}
interface AnimationOption {
    stage?: Stage;
}
export default class Animation extends Eventful {
    stage: Stage;
    private _head;
    private _tail;
    private _running;
    private _time;
    private _pausedTime;
    private _pauseStart;
    private _paused;
    constructor(opts?: AnimationOption);
    addClip(clip: Clip): void;
    addAnimator(animator: Animator<any>): void;
    removeClip(clip: Clip): void;
    removeAnimator(animator: Animator<any>): void;
    update(notTriggerFrameAndStageUpdate?: boolean): void;
    _startLoop(): void;
    start(): void;
    stop(): void;
    pause(): void;
    resume(): void;
    clear(): void;
    isFinished(): boolean;
    animate<T>(target: T, options: {
        loop?: boolean;
    }): Animator<T>;
}
export {};
