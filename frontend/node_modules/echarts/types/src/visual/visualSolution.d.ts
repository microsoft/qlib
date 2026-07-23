import VisualMapping, { VisualMappingOption } from './VisualMapping.js';
import { BuiltinVisualProperty, ParsedValue, DimensionLoose, StageHandlerProgressExecutor } from '../util/types.js';
import SeriesData from '../data/SeriesData.js';
declare type VisualMappingCollection<VisualState extends string> = {
    [key in VisualState]?: {
        [key in BuiltinVisualProperty]?: VisualMapping;
    } & {
        __alphaForOpacity?: VisualMapping;
    };
};
declare type VisualOption = {
    [key in BuiltinVisualProperty]?: any;
};
export declare function createVisualMappings<VisualState extends string>(option: Partial<Record<VisualState, VisualOption>>, stateList: readonly VisualState[], supplementVisualOption: (mappingOption: VisualMappingOption, state: string) => void): VisualMappingCollection<VisualState>;
export declare function replaceVisualOption<T extends string>(thisOption: Partial<Record<T, any>>, newOption: Partial<Record<T, any>>, keys: readonly T[]): void;
/**
 * @param stateList
 * @param visualMappings
 * @param list
 * @param getValueState param: valueOrIndex, return: state.
 * @param scope Scope for getValueState
 * @param dimension Concrete dimension, if used.
 */
export declare function applyVisual<VisualState extends string, Scope>(stateList: readonly VisualState[], visualMappings: VisualMappingCollection<VisualState>, data: SeriesData, getValueState: (this: Scope, valueOrIndex: ParsedValue | number) => VisualState, scope?: Scope, dimension?: DimensionLoose): void;
/**
 * @param data
 * @param stateList
 * @param visualMappings <state, Object.<visualType, module:echarts/visual/VisualMapping>>
 * @param getValueState param: valueOrIndex, return: state.
 * @param dim dimension or dimension index.
 */
export declare function incrementalApplyVisual<VisualState extends string>(stateList: readonly VisualState[], visualMappings: VisualMappingCollection<VisualState>, getValueState: (valueOrIndex: ParsedValue | number) => VisualState, dim?: DimensionLoose): StageHandlerProgressExecutor;
export {};
