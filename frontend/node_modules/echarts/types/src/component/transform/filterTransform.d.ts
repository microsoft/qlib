import { DataTransformOption, ExternalDataTransform } from '../../data/helper/transform.js';
import { ConditionalExpressionOption } from '../../util/conditionalExpression.js';
export interface FilterTransformOption extends DataTransformOption {
    type: 'filter';
    config: ConditionalExpressionOption;
}
export declare const filterTransform: ExternalDataTransform<FilterTransformOption>;
