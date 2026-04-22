import Displayable, { DisplayableProps, CommonStyleProps, DisplayableStatePropNames } from './Displayable';
import Element, { ElementAnimateConfig } from '../Element';
import PathProxy from '../core/PathProxy';
import { PatternObject } from './Pattern';
import { Dictionary, PropType, MapToType } from '../core/types';
import BoundingRect from '../core/BoundingRect';
import { LinearGradientObject } from './LinearGradient';
import { RadialGradientObject } from './RadialGradient';
import Animator from '../animation/Animator';
export interface PathStyleProps extends CommonStyleProps {
    fill?: string | PatternObject | LinearGradientObject | RadialGradientObject;
    stroke?: string | PatternObject | LinearGradientObject | RadialGradientObject;
    decal?: PatternObject;
    strokePercent?: number;
    strokeNoScale?: boolean;
    fillOpacity?: number;
    strokeOpacity?: number;
    lineDash?: false | number[] | 'solid' | 'dashed' | 'dotted';
    lineDashOffset?: number;
    lineWidth?: number;
    lineCap?: CanvasLineCap;
    lineJoin?: CanvasLineJoin;
    miterLimit?: number;
    strokeFirst?: boolean;
}
export declare const DEFAULT_PATH_STYLE: PathStyleProps;
export declare const DEFAULT_PATH_ANIMATION_PROPS: MapToType<PathProps, boolean>;
export interface PathProps extends DisplayableProps {
    strokeContainThreshold?: number;
    segmentIgnoreThreshold?: number;
    subPixelOptimize?: boolean;
    style?: PathStyleProps;
    shape?: Dictionary<any>;
    autoBatch?: boolean;
    __value?: (string | number)[] | (string | number);
    buildPath?: (ctx: PathProxy | CanvasRenderingContext2D, shapeCfg: Dictionary<any>, inBatch?: boolean) => void;
}
declare type PathKey = keyof PathProps;
declare type PathPropertyType = PropType<PathProps, PathKey>;
interface Path<Props extends PathProps = PathProps> {
    animate(key?: '', loop?: boolean): Animator<this>;
    animate(key: 'style', loop?: boolean): Animator<this['style']>;
    animate(key: 'shape', loop?: boolean): Animator<this['shape']>;
    getState(stateName: string): PathState;
    ensureState(stateName: string): PathState;
    states: Dictionary<PathState>;
    stateProxy: (stateName: string) => PathState;
}
export declare type PathStatePropNames = DisplayableStatePropNames | 'shape';
export declare type PathState = Pick<PathProps, PathStatePropNames> & {
    hoverLayer?: boolean;
};
declare class Path<Props extends PathProps = PathProps> extends Displayable<Props> {
    path: PathProxy;
    strokeContainThreshold: number;
    segmentIgnoreThreshold: number;
    subPixelOptimize: boolean;
    style: PathStyleProps;
    autoBatch: boolean;
    private _rectStroke;
    protected _normalState: PathState;
    protected _decalEl: Path;
    shape: Dictionary<any>;
    constructor(opts?: Props);
    update(): void;
    getDecalElement(): Path<PathProps>;
    protected _init(props?: Props): void;
    protected getDefaultStyle(): Props['style'];
    protected getDefaultShape(): {};
    protected canBeInsideText(): boolean;
    protected getInsideTextFill(): "#333" | "#ccc" | "#eee";
    protected getInsideTextStroke(textFill?: string): string;
    buildPath(ctx: PathProxy | CanvasRenderingContext2D, shapeCfg: Dictionary<any>, inBatch?: boolean): void;
    pathUpdated(): void;
    getUpdatedPathProxy(inBatch?: boolean): PathProxy;
    createPathProxy(): void;
    hasStroke(): boolean;
    hasFill(): boolean;
    getBoundingRect(): BoundingRect;
    contain(x: number, y: number): boolean;
    dirtyShape(): void;
    dirty(): void;
    animateShape(loop: boolean): Animator<this["shape"]>;
    updateDuringAnimation(targetKey: string): void;
    attrKV(key: PathKey, value: PathPropertyType): void;
    setShape(obj: Props['shape']): this;
    setShape<T extends keyof Props['shape']>(obj: T, value: Props['shape'][T]): this;
    shapeChanged(): boolean;
    createStyle(obj?: Props['style']): Props["style"];
    protected _innerSaveToNormal(toState: PathState): void;
    protected _applyStateObj(stateName: string, state: PathState, normalState: PathState, keepCurrentStates: boolean, transition: boolean, animationCfg: ElementAnimateConfig): void;
    protected _mergeStates(states: PathState[]): PathState;
    getAnimationStyleProps(): MapToType<PathProps, boolean>;
    isZeroArea(): boolean;
    static extend<Shape extends Dictionary<any>>(defaultProps: {
        type?: string;
        shape?: Shape;
        style?: PathStyleProps;
        beforeBrush?: Displayable['beforeBrush'];
        afterBrush?: Displayable['afterBrush'];
        getBoundingRect?: Displayable['getBoundingRect'];
        calculateTextPosition?: Element['calculateTextPosition'];
        buildPath(this: Path, ctx: CanvasRenderingContext2D | PathProxy, shape: Shape, inBatch?: boolean): void;
        init?(this: Path, opts: PathProps): void;
    }): {
        new (opts?: PathProps & {
            shape: Shape;
        }): Path;
    };
    protected static initDefaultProps: void;
}
export default Path;
