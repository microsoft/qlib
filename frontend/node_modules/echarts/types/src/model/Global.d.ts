/**
 * Caution: If the mechanism should be changed some day, these cases
 * should be considered:
 *
 * (1) In `merge option` mode, if using the same option to call `setOption`
 * many times, the result should be the same (try our best to ensure that).
 * (2) In `merge option` mode, if a component has no id/name specified, it
 * will be merged by index, and the result sequence of the components is
 * consistent to the original sequence.
 * (3) In `replaceMerge` mode, keep the result sequence of the components is
 * consistent to the original sequence, even though there might result in "hole".
 * (4) `reset` feature (in toolbox). Find detailed info in comments about
 * `mergeOption` in module:echarts/model/OptionManager.
 */
import { HashMap } from 'zrender/lib/core/util.js';
import Model from './Model.js';
import ComponentModel from './Component.js';
import SeriesModel from './Series.js';
import { Payload, OptionPreprocessor, ECBasicOption, ECUnitOption, ComponentMainType, ComponentSubType, OptionId, OptionName } from '../util/types.js';
import OptionManager from './OptionManager.js';
import Scheduler from '../core/Scheduler.js';
import { LocaleOption } from '../core/locale.js';
import { PaletteMixin } from './mixin/palette.js';
export interface GlobalModelSetOptionOpts {
    replaceMerge: ComponentMainType | ComponentMainType[];
}
export interface InnerSetOptionOpts {
    replaceMergeMainTypeMap: HashMap<boolean, string>;
}
declare class GlobalModel extends Model<ECUnitOption> {
    option: ECUnitOption;
    private _theme;
    private _locale;
    private _optionManager;
    private _componentsMap;
    /**
     * `_componentsMap` might have "hole" because of remove.
     * So save components count for a certain mainType here.
     */
    private _componentsCount;
    /**
     * Mapping between filtered series list and raw series list.
     * key: filtered series indices, value: raw series indices.
     * Items of `_seriesIndices` never be null/empty/-1.
     * If series has been removed by `replaceMerge`, those series
     * also won't be in `_seriesIndices`, just like be filtered.
     */
    private _seriesIndices;
    /**
     * Key: seriesIndex.
     * Keep consistent with `_seriesIndices`.
     */
    private _seriesIndicesMap;
    /**
     * Model for store update payload
     */
    private _payload;
    scheduler: Scheduler;
    ssr: boolean;
    init(option: ECBasicOption, parentModel: Model, ecModel: GlobalModel, theme: object, locale: object, optionManager: OptionManager): void;
    setOption(option: ECBasicOption, opts: GlobalModelSetOptionOpts, optionPreprocessorFuncs: OptionPreprocessor[]): void;
    /**
     * @param type null/undefined: reset all.
     *        'recreate': force recreate all.
     *        'timeline': only reset timeline option
     *        'media': only reset media query option
     * @return Whether option changed.
     */
    resetOption(type: 'recreate' | 'timeline' | 'media', opt?: Pick<GlobalModelSetOptionOpts, 'replaceMerge'>): boolean;
    private _resetOption;
    mergeOption(option: ECUnitOption): void;
    private _mergeOption;
    /**
     * Get option for output (cloned option and inner info removed)
     */
    getOption(): ECUnitOption;
    getTheme(): Model;
    getLocaleModel(): Model<LocaleOption>;
    setUpdatePayload(payload: Payload): void;
    getUpdatePayload(): Payload;
    /**
     * @param idx If not specified, return the first one.
     */
    getComponent(mainType: ComponentMainType, idx?: number): ComponentModel;
    /**
     * @return Never be null/undefined.
     */
    queryComponents(condition: QueryConditionKindB): ComponentModel[];
    /**
     * The interface is different from queryComponents,
     * which is convenient for inner usage.
     *
     * @usage
     * let result = findComponents(
     *     {mainType: 'dataZoom', query: {dataZoomId: 'abc'}}
     * );
     * let result = findComponents(
     *     {mainType: 'series', subType: 'pie', query: {seriesName: 'uio'}}
     * );
     * let result = findComponents(
     *     {mainType: 'series',
     *     filter: function (model, index) {...}}
     * );
     * // result like [component0, componnet1, ...]
     */
    findComponents(condition: QueryConditionKindA): ComponentModel[];
    /**
     * Travel components (before filtered).
     *
     * @usage
     * eachComponent('legend', function (legendModel, index) {
     *     ...
     * });
     * eachComponent(function (componentType, model, index) {
     *     // componentType does not include subType
     *     // (componentType is 'a' but not 'a.b')
     * });
     * eachComponent(
     *     {mainType: 'dataZoom', query: {dataZoomId: 'abc'}},
     *     function (model, index) {...}
     * );
     * eachComponent(
     *     {mainType: 'series', subType: 'pie', query: {seriesName: 'uio'}},
     *     function (model, index) {...}
     * );
     */
    eachComponent<T>(cb: EachComponentAllCallback, context?: T): void;
    eachComponent<T>(mainType: string, cb: EachComponentInMainTypeCallback, context?: T): void;
    eachComponent<T>(mainType: QueryConditionKindA, cb: EachComponentInMainTypeCallback, context?: T): void;
    /**
     * Get series list before filtered by name.
     */
    getSeriesByName(name: OptionName): SeriesModel[];
    /**
     * Get series list before filtered by index.
     */
    getSeriesByIndex(seriesIndex: number): SeriesModel;
    /**
     * Get series list before filtered by type.
     * FIXME: rename to getRawSeriesByType?
     */
    getSeriesByType(subType: ComponentSubType): SeriesModel[];
    /**
     * Get all series before filtered.
     */
    getSeries(): SeriesModel[];
    /**
     * Count series before filtered.
     */
    getSeriesCount(): number;
    /**
     * After filtering, series may be different
     * from raw series.
     */
    eachSeries<T>(cb: (this: T, series: SeriesModel, rawSeriesIndex: number) => void, context?: T): void;
    /**
     * Iterate raw series before filtered.
     *
     * @param {Function} cb
     * @param {*} context
     */
    eachRawSeries<T>(cb: (this: T, series: SeriesModel, rawSeriesIndex: number) => void, context?: T): void;
    /**
     * After filtering, series may be different.
     * from raw series.
     */
    eachSeriesByType<T>(subType: ComponentSubType, cb: (this: T, series: SeriesModel, rawSeriesIndex: number) => void, context?: T): void;
    /**
     * Iterate raw series before filtered of given type.
     */
    eachRawSeriesByType<T>(subType: ComponentSubType, cb: (this: T, series: SeriesModel, rawSeriesIndex: number) => void, context?: T): void;
    isSeriesFiltered(seriesModel: SeriesModel): boolean;
    getCurrentSeriesIndices(): number[];
    filterSeries<T>(cb: (this: T, series: SeriesModel, rawSeriesIndex: number) => boolean, context?: T): void;
    restoreData(payload?: Payload): void;
    private static internalField;
}
/**
 * @param condition.mainType Mandatory.
 * @param condition.subType Optional.
 * @param condition.query like {xxxIndex, xxxId, xxxName},
 *        where xxx is mainType.
 *        If query attribute is null/undefined or has no index/id/name,
 *        do not filtering by query conditions, which is convenient for
 *        no-payload situations or when target of action is global.
 * @param condition.filter parameter: component, return boolean.
 */
export interface QueryConditionKindA {
    mainType: ComponentMainType;
    subType?: ComponentSubType;
    query?: {
        [k: string]: number | number[] | string | string[];
    };
    filter?: (cmpt: ComponentModel) => boolean;
}
/**
 * If none of index and id and name used, return all components with mainType.
 * @param condition.mainType
 * @param condition.subType If ignore, only query by mainType
 * @param condition.index Either input index or id or name.
 * @param condition.id Either input index or id or name.
 * @param condition.name Either input index or id or name.
 */
export interface QueryConditionKindB {
    mainType: ComponentMainType;
    subType?: ComponentSubType;
    index?: number | number[];
    id?: OptionId | OptionId[];
    name?: OptionName | OptionName[];
}
export interface EachComponentAllCallback {
    (mainType: string, model: ComponentModel, componentIndex: number): void;
}
interface EachComponentInMainTypeCallback {
    (model: ComponentModel, componentIndex: number): void;
}
interface GlobalModel extends PaletteMixin<ECUnitOption> {
}
export default GlobalModel;
