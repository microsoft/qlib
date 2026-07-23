import Axis from '../../coord/Axis.js';
import Scale from '../../scale/Scale.js';
import TimelineModel from './TimelineModel.js';
import { LabelOption } from '../../util/types.js';
import Model from '../../model/Model.js';
/**
 * Extend axis 2d
 */
declare class TimelineAxis extends Axis {
    type: 'category' | 'time' | 'value';
    model: TimelineModel;
    constructor(dim: string, scale: Scale, coordExtent: [number, number], axisType: 'category' | 'time' | 'value');
    /**
     * @override
     */
    getLabelModel(): Model<LabelOption>;
    /**
     * @override
     */
    isHorizontal(): boolean;
}
export default TimelineAxis;
