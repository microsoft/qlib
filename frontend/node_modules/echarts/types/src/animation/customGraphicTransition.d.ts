import Element from 'zrender/lib/Element.js';
import Model from '../model/Model.js';
import { AnimationOption, AnimationOptionMixin } from '../util/types.js';
import { Dictionary } from 'zrender/lib/core/types.js';
import { PathStyleProps } from 'zrender/lib/graphic/Path.js';
import { TransformProp } from 'zrender/lib/core/Transformable.js';
export declare const ELEMENT_ANIMATABLE_PROPS: readonly ["", "style", "shape", "extra"];
export declare type TransitionProps = string | string[];
export declare type ElementRootTransitionProp = TransformProp | 'shape' | 'extra' | 'style';
export interface TransitionOptionMixin<T = Record<string, any>> {
    transition?: (keyof T & string) | ((keyof T & string)[]) | 'all';
    enterFrom?: T;
    leaveTo?: T;
    enterAnimation?: AnimationOption;
    updateAnimation?: AnimationOption;
    leaveAnimation?: AnimationOption;
}
declare type TransitionElementOption = Partial<Record<TransformProp, number>> & {
    shape?: Dictionary<any> & TransitionOptionMixin;
    style?: PathStyleProps & TransitionOptionMixin;
    extra?: Dictionary<any> & TransitionOptionMixin;
    invisible?: boolean;
    silent?: boolean;
    autoBatch?: boolean;
    ignore?: boolean;
    during?: (params: TransitionDuringAPI) => void;
} & TransitionOptionMixin;
export interface TransitionBaseDuringAPI {
    setTransform(key: TransformProp, val: number): this;
    getTransform(key: TransformProp): number;
    setExtra(key: string, val: unknown): this;
    getExtra(key: string): unknown;
}
export interface TransitionDuringAPI<StyleOpt extends any = any, ShapeOpt extends any = any> extends TransitionBaseDuringAPI {
    setShape<T extends keyof ShapeOpt>(key: T, val: ShapeOpt[T]): this;
    getShape<T extends keyof ShapeOpt>(key: T): ShapeOpt[T];
    setStyle<T extends keyof StyleOpt>(key: T, val: StyleOpt[T]): this;
    getStyle<T extends keyof StyleOpt>(key: T): StyleOpt[T];
}
export declare function applyUpdateTransition(el: Element, elOption: TransitionElementOption, animatableModel?: Model<AnimationOptionMixin>, opts?: {
    dataIndex?: number;
    isInit?: boolean;
    clearStyle?: boolean;
}): void;
export declare function updateLeaveTo(el: Element, elOption: TransitionElementOption): void;
export declare function applyLeaveTransition(el: Element, elOption: TransitionElementOption, animatableModel: Model<AnimationOptionMixin>, onRemove?: () => void): void;
export declare function isTransitionAll(transition: TransitionProps): transition is 'all';
export {};
