import SeriesModel from '../../model/Series.js';
import { StageHandlerPlanReturn } from '../../util/types.js';
/**
 * @return {string} If large mode changed, return string 'reset';
 */
export default function createRenderPlanner(): (seriesModel: SeriesModel) => StageHandlerPlanReturn;
