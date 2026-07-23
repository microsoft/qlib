import { BrushControllerEvents } from '../helper/BrushController.js';
import ComponentView from '../../view/Component.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import GlobalModel from '../../model/Global.js';
import ParallelAxisModel, { ParallelAreaSelectStyleProps } from '../../coord/parallel/AxisModel.js';
import { Payload } from '../../util/types.js';
import ParallelModel from '../../coord/parallel/ParallelModel.js';
import { ParallelAxisLayoutInfo } from '../../coord/parallel/Parallel.js';
declare class ParallelAxisView extends ComponentView {
    static type: string;
    readonly type: string;
    private _brushController;
    private _axisGroup;
    axisModel: ParallelAxisModel;
    api: ExtensionAPI;
    init(ecModel: GlobalModel, api: ExtensionAPI): void;
    render(axisModel: ParallelAxisModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    _refreshBrushController(builderOpt: Pick<ParallelAxisLayoutInfo, 'position' | 'rotation'>, areaSelectStyle: ParallelAreaSelectStyleProps, axisModel: ParallelAxisModel, coordSysModel: ParallelModel, areaWidth: ParallelAreaSelectStyleProps['width'], api: ExtensionAPI): void;
    _onBrush(eventParam: BrushControllerEvents['brush']): void;
    dispose(): void;
}
export default ParallelAxisView;
