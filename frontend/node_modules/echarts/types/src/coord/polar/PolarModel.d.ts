import { ComponentOption, CircleLayoutOptionMixin } from '../../util/types.js';
import ComponentModel from '../../model/Component.js';
import Polar from './Polar.js';
import { AngleAxisModel, RadiusAxisModel } from './AxisModel.js';
export interface PolarOption extends ComponentOption, CircleLayoutOptionMixin {
    mainType?: 'polar';
}
declare class PolarModel extends ComponentModel<PolarOption> {
    static type: "polar";
    type: "polar";
    static dependencies: string[];
    coordinateSystem: Polar;
    findAxisModel(axisType: 'angleAxis'): AngleAxisModel;
    findAxisModel(axisType: 'radiusAxis'): RadiusAxisModel;
    static defaultOption: PolarOption;
}
export default PolarModel;
