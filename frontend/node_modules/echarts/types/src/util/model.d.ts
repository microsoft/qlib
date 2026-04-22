import { HashMap } from 'zrender/lib/core/util.js';
import GlobalModel from '../model/Global.js';
import ComponentModel, { ComponentModelConstructor } from '../model/Component.js';
import SeriesData from '../data/SeriesData.js';
import { ComponentOption, ComponentMainType, ComponentSubType, DisplayStateHostOption, OptionDataItem, OptionDataValue, TooltipRenderMode, Payload, OptionId, InterpolatableValue } from './types.js';
import SeriesModel from '../model/Series.js';
import CartesianAxisModel from '../coord/cartesian/AxisModel.js';
import GridModel from '../coord/cartesian/GridModel.js';
/**
 * If value is not array, then translate it to array.
 * @param  {*} value
 * @return {Array} [value] or value
 */
export declare function normalizeToArray<T>(value?: T | T[]): T[];
/**
 * Sync default option between normal and emphasis like `position` and `show`
 * In case some one will write code like
 *     label: {
 *          show: false,
 *          position: 'outside',
 *          fontSize: 18
 *     },
 *     emphasis: {
 *          label: { show: true }
 *     }
 */
export declare function defaultEmphasis(opt: DisplayStateHostOption, key: string, subOpts: string[]): void;
export declare const TEXT_STYLE_OPTIONS: readonly ["fontStyle", "fontWeight", "fontSize", "fontFamily", "rich", "tag", "color", "textBorderColor", "textBorderWidth", "width", "height", "lineHeight", "align", "verticalAlign", "baseline", "shadowColor", "shadowBlur", "shadowOffsetX", "shadowOffsetY", "textShadowColor", "textShadowBlur", "textShadowOffsetX", "textShadowOffsetY", "backgroundColor", "borderColor", "borderWidth", "borderRadius", "padding"];
/**
 * The method does not ensure performance.
 * data could be [12, 2323, {value: 223}, [1221, 23], {value: [2, 23]}]
 * This helper method retrieves value from data.
 */
export declare function getDataItemValue(dataItem: OptionDataItem): OptionDataValue | OptionDataValue[];
/**
 * data could be [12, 2323, {value: 223}, [1221, 23], {value: [2, 23]}]
 * This helper method determine if dataItem has extra option besides value
 */
export declare function isDataItemOption(dataItem: OptionDataItem): boolean;
export interface MappingExistingItem {
    id?: OptionId;
    name?: string;
}
/**
 * The array `MappingResult<T>[]` exactly represents the content of the result
 * components array after merge.
 * The indices are the same as the `existings`.
 * Items will not be `null`/`undefined` even if the corresponding `existings` will be removed.
 */
declare type MappingResult<T> = MappingResultItem<T>[];
interface MappingResultItem<T extends MappingExistingItem = MappingExistingItem> {
    existing: T;
    newOption: ComponentOption;
    brandNew: boolean;
    keyInfo: {
        name: string;
        id: string;
        mainType: ComponentMainType;
        subType: ComponentSubType;
    };
}
declare type MappingToExistsMode = 'normalMerge' | 'replaceMerge' | 'replaceAll';
/**
 * Mapping to existings for merge.
 *
 * Mode "normalMege":
 *     The mapping result (merge result) will keep the order of the existing
 *     component, rather than the order of new option. Because we should ensure
 *     some specified index reference (like xAxisIndex) keep work.
 *     And in most cases, "merge option" is used to update partial option but not
 *     be expected to change the order.
 *
 * Mode "replaceMege":
 *     (1) Only the id mapped components will be merged.
 *     (2) Other existing components (except internal components) will be removed.
 *     (3) Other new options will be used to create new component.
 *     (4) The index of the existing components will not be modified.
 *     That means their might be "hole" after the removal.
 *     The new components are created first at those available index.
 *
 * Mode "replaceAll":
 *     This mode try to support that reproduce an echarts instance from another
 *     echarts instance (via `getOption`) in some simple cases.
 *     In this scenario, the `result` index are exactly the consistent with the `newCmptOptions`,
 *     which ensures the component index referring (like `xAxisIndex: ?`) corrent. That is,
 *     the "hole" in `newCmptOptions` will also be kept.
 *     On the contrary, other modes try best to eliminate holes.
 *     PENDING: This is an experimental mode yet.
 *
 * @return See the comment of <MappingResult>.
 */
export declare function mappingToExists<T extends MappingExistingItem>(existings: T[], newCmptOptions: ComponentOption[], mode: MappingToExistsMode): MappingResult<T>;
export declare function convertOptionIdName(idOrName: unknown, defaultValue: string): string;
export declare function isNameSpecified(componentModel: ComponentModel): boolean;
/**
 * @public
 * @param {Object} cmptOption
 * @return {boolean}
 */
export declare function isComponentIdInternal(cmptOption: {
    id?: MappingExistingItem['id'];
}): boolean;
export declare function makeInternalComponentId(idSuffix: string): string;
export declare function setComponentTypeToKeyInfo(mappingResult: MappingResult<MappingExistingItem & {
    subType?: ComponentSubType;
}>, mainType: ComponentMainType, componentModelCtor: ComponentModelConstructor): void;
declare type BatchItem = {
    seriesId: OptionId;
    dataIndex: number | number[];
};
/**
 * A helper for removing duplicate items between batchA and batchB,
 * and in themselves, and categorize by series.
 *
 * @param batchA Like: [{seriesId: 2, dataIndex: [32, 4, 5]}, ...]
 * @param batchB Like: [{seriesId: 2, dataIndex: [32, 4, 5]}, ...]
 * @return result: [resultBatchA, resultBatchB]
 */
export declare function compressBatches(batchA: BatchItem[], batchB: BatchItem[]): [BatchItem[], BatchItem[]];
/**
 * @param payload Contains dataIndex (means rawIndex) / dataIndexInside / name
 *                         each of which can be Array or primary type.
 * @return dataIndex If not found, return undefined/null.
 */
export declare function queryDataIndex(data: SeriesData, payload: Payload & {
    dataIndexInside?: number | number[];
    dataIndex?: number | number[];
    name?: string | string[];
}): number | number[];
/**
 * Enable property storage to any host object.
 * Notice: Serialization is not supported.
 *
 * For example:
 * let inner = zrUitl.makeInner();
 *
 * function some1(hostObj) {
 *      inner(hostObj).someProperty = 1212;
 *      ...
 * }
 * function some2() {
 *      let fields = inner(this);
 *      fields.someProperty1 = 1212;
 *      fields.someProperty2 = 'xx';
 *      ...
 * }
 *
 * @return {Function}
 */
export declare function makeInner<T, Host extends object>(): (hostObj: Host) => T;
/**
 * If string, e.g., 'geo', means {geoIndex: 0}.
 * If Object, could contain some of these properties below:
 * {
 *     seriesIndex, seriesId, seriesName,
 *     geoIndex, geoId, geoName,
 *     bmapIndex, bmapId, bmapName,
 *     xAxisIndex, xAxisId, xAxisName,
 *     yAxisIndex, yAxisId, yAxisName,
 *     gridIndex, gridId, gridName,
 *     ... (can be extended)
 * }
 * Each properties can be number|string|Array.<number>|Array.<string>
 * For example, a finder could be
 * {
 *     seriesIndex: 3,
 *     geoId: ['aa', 'cc'],
 *     gridName: ['xx', 'rr']
 * }
 * xxxIndex can be set as 'all' (means all xxx) or 'none' (means not specify)
 * If nothing or null/undefined specified, return nothing.
 * If both `abcIndex`, `abcId`, `abcName` specified, only one work.
 * The priority is: index > id > name, the same with `ecModel.queryComponents`.
 */
export declare type ModelFinderIndexQuery = number | number[] | 'all' | 'none' | false;
export declare type ModelFinderIdQuery = OptionId | OptionId[];
export declare type ModelFinderNameQuery = OptionId | OptionId[];
export declare type ModelFinder = string | ModelFinderObject;
export declare type ModelFinderObject = {
    seriesIndex?: ModelFinderIndexQuery;
    seriesId?: ModelFinderIdQuery;
    seriesName?: ModelFinderNameQuery;
    geoIndex?: ModelFinderIndexQuery;
    geoId?: ModelFinderIdQuery;
    geoName?: ModelFinderNameQuery;
    bmapIndex?: ModelFinderIndexQuery;
    bmapId?: ModelFinderIdQuery;
    bmapName?: ModelFinderNameQuery;
    xAxisIndex?: ModelFinderIndexQuery;
    xAxisId?: ModelFinderIdQuery;
    xAxisName?: ModelFinderNameQuery;
    yAxisIndex?: ModelFinderIndexQuery;
    yAxisId?: ModelFinderIdQuery;
    yAxisName?: ModelFinderNameQuery;
    gridIndex?: ModelFinderIndexQuery;
    gridId?: ModelFinderIdQuery;
    gridName?: ModelFinderNameQuery;
    dataIndex?: number;
    dataIndexInside?: number;
};
/**
 * {
 *     seriesModels: [seriesModel1, seriesModel2],
 *     seriesModel: seriesModel1, // The first model
 *     geoModels: [geoModel1, geoModel2],
 *     geoModel: geoModel1, // The first model
 *     ...
 * }
 */
export declare type ParsedModelFinder = {
    [key: string]: ComponentModel | ComponentModel[] | undefined;
};
export declare type ParsedModelFinderKnown = ParsedModelFinder & {
    seriesModels?: SeriesModel[];
    seriesModel?: SeriesModel;
    xAxisModels?: CartesianAxisModel[];
    xAxisModel?: CartesianAxisModel;
    yAxisModels?: CartesianAxisModel[];
    yAxisModel?: CartesianAxisModel;
    gridModels?: GridModel[];
    gridModel?: GridModel;
    dataIndex?: number;
    dataIndexInside?: number;
};
/**
 * The same behavior as `component.getReferringComponents`.
 */
export declare function parseFinder(ecModel: GlobalModel, finderInput: ModelFinder, opt?: {
    defaultMainType?: ComponentMainType;
    includeMainTypes?: ComponentMainType[];
    enableAll?: boolean;
    enableNone?: boolean;
}): ParsedModelFinder;
export declare function preParseFinder(finderInput: ModelFinder, opt?: {
    includeMainTypes?: ComponentMainType[];
}): {
    mainTypeSpecified: boolean;
    queryOptionMap: HashMap<QueryReferringUserOption, ComponentMainType>;
    others: Partial<Pick<ParsedModelFinderKnown, 'dataIndex' | 'dataIndexInside'>>;
};
export declare type QueryReferringUserOption = {
    index?: ModelFinderIndexQuery;
    id?: ModelFinderIdQuery;
    name?: ModelFinderNameQuery;
};
export declare const SINGLE_REFERRING: QueryReferringOpt;
export declare const MULTIPLE_REFERRING: QueryReferringOpt;
export declare type QueryReferringOpt = {
    useDefault?: boolean;
    enableAll?: boolean;
    enableNone?: boolean;
};
export declare function queryReferringComponents(ecModel: GlobalModel, mainType: ComponentMainType, userOption: QueryReferringUserOption, opt?: QueryReferringOpt): {
    models: ComponentModel[];
    specified: boolean;
};
export declare function setAttribute(dom: HTMLElement, key: string, value: any): void;
export declare function getAttribute(dom: HTMLElement, key: string): any;
export declare function getTooltipRenderMode(renderModeOption: TooltipRenderMode | 'auto'): TooltipRenderMode;
/**
 * Group a list by key.
 */
export declare function groupData<T, R extends string | number>(array: T[], getKey: (item: T) => R): {
    keys: R[];
    buckets: HashMap<T[], R>;
};
/**
 * Interpolate raw values of a series with percent
 *
 * @param data         data
 * @param labelModel   label model of the text element
 * @param sourceValue  start value. May be null/undefined when init.
 * @param targetValue  end value
 * @param percent      0~1 percentage; 0 uses start value while 1 uses end value
 * @return             interpolated values
 *                     If `sourceValue` and `targetValue` are `number`, return `number`.
 *                     If `sourceValue` and `targetValue` are `string`, return `string`.
 *                     If `sourceValue` and `targetValue` are `(string | number)[]`, return `(string | number)[]`.
 *                     Other cases do not supported.
 */
export declare function interpolateRawValues(data: SeriesData, precision: number | 'auto', sourceValue: InterpolatableValue, targetValue: InterpolatableValue, percent: number): InterpolatableValue;
export {};
