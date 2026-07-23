import { DimensionDefinitionLoose, SourceFormat, DimensionDefinition, DimensionIndex, OptionDataValue, DimensionLoose, ParsedValue, OptionSourceDataObjectRows, OptionSourceDataArrayRows } from '../../util/types.js';
import { Source } from '../Source.js';
export declare type PipedDataTransformOption = DataTransformOption[];
export declare type DataTransformType = string;
export declare type DataTransformConfig = unknown;
export interface DataTransformOption {
    type: DataTransformType;
    config?: DataTransformConfig;
    print?: boolean;
}
export interface ExternalDataTransform<TO extends DataTransformOption = DataTransformOption> {
    type: string;
    __isBuiltIn?: boolean;
    transform: (param: ExternalDataTransformParam<TO>) => ExternalDataTransformResultItem | ExternalDataTransformResultItem[];
}
interface ExternalDataTransformParam<TO extends DataTransformOption = DataTransformOption> {
    upstream: ExternalSource;
    upstreamList: ExternalSource[];
    config: TO['config'];
}
export interface ExternalDataTransformResultItem {
    /**
     * If `data` is null/undefined, inherit upstream data.
     */
    data: OptionSourceDataArrayRows | OptionSourceDataObjectRows;
    /**
     * A `transform` can optionally return a dimensions definition.
     * The rule:
     * If this `transform result` have different dimensions from the upstream, it should return
     * a new dimension definition. For example, this transform inherit the upstream data totally
     * but add a extra dimension.
     * Otherwise, do not need to return that dimension definition. echarts will inherit dimension
     * definition from the upstream.
     */
    dimensions?: DimensionDefinitionLoose[];
}
export declare type DataTransformDataItem = ExternalDataTransformResultItem['data'][number];
export interface ExternalDimensionDefinition extends Partial<DimensionDefinition> {
    index: DimensionIndex;
}
/**
 * TODO: disable writable.
 * This structure will be exposed to users.
 */
export declare class ExternalSource {
    /**
     * [Caveat]
     * This instance is to be exposed to users.
     * (1) DO NOT mount private members on this instance directly.
     * If we have to use private members, we can make them in closure or use `makeInner`.
     * (2) "source header count" is not provided to transform, because it's complicated to manage
     * header and dimensions definition in each transform. Source headers are all normalized to
     * dimensions definitions in transforms and their downstreams.
     */
    sourceFormat: SourceFormat;
    getRawData(): Source['data'];
    getRawDataItem(dataIndex: number): DataTransformDataItem;
    cloneRawData(): Source['data'];
    /**
     * @return If dimension not found, return null/undefined.
     */
    getDimensionInfo(dim: DimensionLoose): ExternalDimensionDefinition;
    /**
     * dimensions defined if and only if either:
     * (a) dataset.dimensions are declared.
     * (b) dataset data include dimensions definitions in data (detected or via specified `sourceHeader`).
     * If dimensions are defined, `dimensionInfoAll` is corresponding to
     * the defined dimensions.
     * Otherwise, `dimensionInfoAll` is determined by data columns.
     * @return Always return an array (even empty array).
     */
    cloneAllDimensionInfo(): ExternalDimensionDefinition[];
    count(): number;
    /**
     * Only support by dimension index.
     * No need to support by dimension name in transform function,
     * because transform function is not case-specific, no need to use name literally.
     */
    retrieveValue(dataIndex: number, dimIndex: DimensionIndex): OptionDataValue;
    retrieveValueFromItem(dataItem: DataTransformDataItem, dimIndex: DimensionIndex): OptionDataValue;
    convertValue(rawVal: unknown, dimInfo: ExternalDimensionDefinition): ParsedValue;
}
export declare function registerExternalTransform(externalTransform: ExternalDataTransform): void;
export declare function applyDataTransform(rawTransOption: DataTransformOption | PipedDataTransformOption, sourceList: Source[], infoForPrint: {
    datasetIndex: number;
}): Source[];
export {};
