/**
 * ECharts option manager
 */
import ExtensionAPI from '../core/ExtensionAPI.js';
import { OptionPreprocessor, ECUnitOption, ECBasicOption } from '../util/types.js';
import GlobalModel, { InnerSetOptionOpts } from './Global.js';
/**
 * TERM EXPLANATIONS:
 * See `ECOption` and `ECUnitOption` in `src/util/types.ts`.
 */
declare class OptionManager {
    private _api;
    private _timelineOptions;
    private _mediaList;
    private _mediaDefault;
    /**
     * -1, means default.
     * empty means no media.
     */
    private _currentMediaIndices;
    private _optionBackup;
    private _newBaseOption;
    constructor(api: ExtensionAPI);
    setOption(rawOption: ECBasicOption, optionPreprocessorFuncs: OptionPreprocessor[], opt: InnerSetOptionOpts): void;
    mountOption(isRecreate: boolean): ECUnitOption;
    getTimelineOption(ecModel: GlobalModel): ECUnitOption;
    getMediaOption(ecModel: GlobalModel): ECUnitOption[];
}
/**
 * Consider case:
 * `chart.setOption(opt1);`
 * Then user do some interaction like dataZoom, dataView changing.
 * `chart.setOption(opt2);`
 * Then user press 'reset button' in toolbox.
 *
 * After doing that all of the interaction effects should be reset, the
 * chart should be the same as the result of invoke
 * `chart.setOption(opt1); chart.setOption(opt2);`.
 *
 * Although it is not able ensure that
 * `chart.setOption(opt1); chart.setOption(opt2);` is equivalents to
 * `chart.setOption(merge(opt1, opt2));` exactly,
 * this might be the only simple way to implement that feature.
 *
 * MEMO: We've considered some other approaches:
 * 1. Each model handles its self restoration but not uniform treatment.
 *     (Too complex in logic and error-prone)
 * 2. Use a shadow ecModel. (Performance expensive)
 *
 * FIXME: A possible solution:
 * Add a extra level of model for each component model. The inheritance chain would be:
 * ecModel <- componentModel <- componentActionModel <- dataItemModel
 * And all of the actions can only modify the `componentActionModel` rather than
 * `componentModel`. `setOption` will only modify the `ecModel` and `componentModel`.
 * When "resotre" action triggered, model from `componentActionModel` will be discarded
 * instead of recreating the "ecModel" from the "_optionBackup".
 */
export default OptionManager;
