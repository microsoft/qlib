import { DatasetModel } from '../../component/dataset/install.js';
import SeriesModel from '../../model/Series.js';
import { Source } from '../Source.js';
import DataStore from '../DataStore.js';
import { SeriesDataSchema } from './SeriesDataSchema.js';
/**
 * [REQUIREMENT_MEMO]:
 * (0) `metaRawOption` means `dimensions`/`sourceHeader`/`seriesLayoutBy` in raw option.
 * (1) Keep support the feature: `metaRawOption` can be specified both on `series` and
 * `root-dataset`. Them on `series` has higher priority.
 * (2) Do not support to set `metaRawOption` on a `non-root-dataset`, because it might
 * confuse users: whether those props indicate how to visit the upstream source or visit
 * the transform result source, and some transforms has nothing to do with these props,
 * and some transforms might have multiple upstream.
 * (3) Transforms should specify `metaRawOption` in each output, just like they can be
 * declared in `root-dataset`.
 * (4) At present only support visit source in `SERIES_LAYOUT_BY_COLUMN` in transforms.
 * That is for reducing complexity in transforms.
 * PENDING: Whether to provide transposition transform?
 *
 * [IMPLEMENTAION_MEMO]:
 * "sourceVisitConfig" are calculated from `metaRawOption` and `data`.
 * They will not be calculated until `source` is about to be visited (to prevent from
 * duplicate calcuation). `source` is visited only in series and input to transforms.
 *
 * [DIMENSION_INHERIT_RULE]:
 * By default the dimensions are inherited from ancestors, unless a transform return
 * a new dimensions definition.
 * Consider the case:
 * ```js
 * dataset: [{
 *     source: [ ['Product', 'Sales', 'Prise'], ['Cookies', 321, 44.21], ...]
 * }, {
 *     transform: { type: 'filter', ... }
 * }]
 * dataset: [{
 *     dimension: ['Product', 'Sales', 'Prise'],
 *     source: [ ['Cookies', 321, 44.21], ...]
 * }, {
 *     transform: { type: 'filter', ... }
 * }]
 * ```
 * The two types of option should have the same behavior after transform.
 *
 *
 * [SCENARIO]:
 * (1) Provide source data directly:
 * ```js
 * series: {
 *     encode: {...},
 *     dimensions: [...]
 *     seriesLayoutBy: 'row',
 *     data: [[...]]
 * }
 * ```
 * (2) Series refer to dataset.
 * ```js
 * series: [{
 *     encode: {...}
 *     // Ignore datasetIndex means `datasetIndex: 0`
 *     // and the dimensions defination in dataset is used
 * }, {
 *     encode: {...},
 *     seriesLayoutBy: 'column',
 *     datasetIndex: 1
 * }]
 * ```
 * (3) dataset transform
 * ```js
 * dataset: [{
 *     source: [...]
 * }, {
 *     source: [...]
 * }, {
 *     // By default from 0.
 *     transform: { type: 'filter', config: {...} }
 * }, {
 *     // Piped.
 *     transform: [
 *         { type: 'filter', config: {...} },
 *         { type: 'sort', config: {...} }
 *     ]
 * }, {
 *     id: 'regressionData',
 *     fromDatasetIndex: 1,
 *     // Third-party transform
 *     transform: { type: 'ecStat:regression', config: {...} }
 * }, {
 *     // retrieve the extra result.
 *     id: 'regressionFormula',
 *     fromDatasetId: 'regressionData',
 *     fromTransformResult: 1
 * }]
 * ```
 */
export declare class SourceManager {
    private _sourceHost;
    private _sourceList;
    private _storeList;
    private _upstreamSignList;
    private _versionSignBase;
    private _dirty;
    constructor(sourceHost: DatasetModel | SeriesModel);
    /**
     * Mark dirty.
     */
    dirty(): void;
    private _setLocalSource;
    /**
     * For detecting whether the upstream source is dirty, so that
     * the local cached source (in `_sourceList`) should be discarded.
     */
    private _getVersionSign;
    /**
     * Always return a source instance. Otherwise throw error.
     */
    prepareSource(): void;
    private _createSource;
    private _applyTransform;
    private _isDirty;
    /**
     * @param sourceIndex By default 0, means "main source".
     *                    In most cases there is only one source.
     */
    getSource(sourceIndex?: number): Source;
    /**
     *
     * Get a data store which can be shared across series.
     * Only available for series.
     *
     * @param seriesDimRequest Dimensions that are generated in series.
     *        Should have been sorted by `storeDimIndex` asc.
     */
    getSharedDataStore(seriesDimRequest: SeriesDataSchema): DataStore;
    private _innerGetDataStore;
    /**
     * PENDING: Is it fast enough?
     * If no upstream, return empty array.
     */
    private _getUpstreamSourceManagers;
    private _getSourceMetaRawOption;
}
export declare function disableTransformOptionMerge(datasetModel: DatasetModel): void;
