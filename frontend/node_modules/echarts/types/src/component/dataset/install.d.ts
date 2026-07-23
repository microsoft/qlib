/**
 * This module is imported by echarts directly.
 *
 * Notice:
 * Always keep this file exists for backward compatibility.
 * Because before 4.1.0, dataset is an optional component,
 * some users may import this module manually.
 */
import ComponentModel from '../../model/Component.js';
import { ComponentOption, SeriesEncodeOptionMixin, OptionSourceData, SeriesLayoutBy, OptionSourceHeader } from '../../util/types.js';
import { DataTransformOption, PipedDataTransformOption } from '../../data/helper/transform.js';
import GlobalModel from '../../model/Global.js';
import Model from '../../model/Model.js';
import { SourceManager } from '../../data/helper/sourceManager.js';
import { EChartsExtensionInstallRegisters } from '../../extension.js';
export interface DatasetOption extends Pick<ComponentOption, 'type' | 'id' | 'name'>, Pick<SeriesEncodeOptionMixin, 'dimensions'> {
    mainType?: 'dataset';
    seriesLayoutBy?: SeriesLayoutBy;
    sourceHeader?: OptionSourceHeader;
    source?: OptionSourceData;
    fromDatasetIndex?: number;
    fromDatasetId?: string;
    transform?: DataTransformOption | PipedDataTransformOption;
    fromTransformResult?: number;
}
export declare class DatasetModel<Opts extends DatasetOption = DatasetOption> extends ComponentModel<Opts> {
    type: string;
    static type: string;
    static defaultOption: DatasetOption;
    private _sourceManager;
    init(option: Opts, parentModel: Model, ecModel: GlobalModel): void;
    mergeOption(newOption: Opts, ecModel: GlobalModel): void;
    optionUpdated(): void;
    getSourceManager(): SourceManager;
}
export declare function install(registers: EChartsExtensionInstallRegisters): void;
