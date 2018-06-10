/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {ENV} from '../environment';
import {keep, tidy} from '../globals';
import {scalar, zerosLike} from '../ops/ops';
// tslint:disable-next-line:max-line-length
import {ConfigDict, Serializable, SerializableConstructor, SerializationMap} from '../serialization';
import {Scalar} from '../tensor';
import {NamedVariableMap} from '../types';

import {Optimizer} from './optimizer';

/** @doclink Optimizer */
export class AMSgradOptimizer extends Optimizer {
  static className = 'AMSgradOptimizer';
  private c: Scalar;
  private epsScalar: Scalar;
  private beta1Scalar: Scalar;
  private beta2Scalar: Scalar;
  private oneMinusBeta1: Scalar;
  private oneMinusBeta2: Scalar;
  private prevFirstMoment: NamedVariableMap = {};
  private prevSecondMoment: NamedVariableMap = {};

  constructor(
      protected learningRate: number, protected beta1: number,
      protected beta2: number, protected epsilon = 1e-8) {
    super();
    this.c = keep(scalar(-learningRate));
    this.epsScalar = keep(scalar(epsilon));
    this.beta1Scalar = keep(scalar(beta1));
    this.beta2Scalar = keep(scalar(beta2));
    this.oneMinusBeta1 = keep(scalar(1 - beta1));
    this.oneMinusBeta2 = keep(scalar(1 - beta2));
  }

  applyGradients(variableGradients: NamedVariableMap) {
    tidy(() => {
      for (const variableName in variableGradients) {
        const value = ENV.engine.registeredVariables[variableName];
        if (this.prevFirstMoment[variableName] == null) {
          const trainable = false;
          this.prevFirstMoment[variableName] =
              zerosLike(value).variable(trainable);
        }
        if (this.prevSecondMoment[variableName] == null) {
          const trainable = false;
          this.prevSecondMoment[variableName] =
              zerosLike(value).variable(trainable);
        }

        const gradient = variableGradients[variableName];
        const firstMoment = this.prevFirstMoment[variableName];
        const secondMoment = this.prevSecondMoment[variableName];

        const newFirstMoment = this.beta1Scalar.mul(firstMoment)
                                   .add(this.oneMinusBeta1.mul(gradient));
        let newSecondMoment =
            this.beta2Scalar.mul(secondMoment)
                .add(this.oneMinusBeta2.mul(gradient.square()));

        newSecondMoment = newSecondMoment.maximum(secondMoment);

        this.prevFirstMoment[variableName].assign(newFirstMoment);
        this.prevSecondMoment[variableName].assign(newSecondMoment);

        const newValue = this.c
                             .mul(newFirstMoment.div(
                                 this.epsScalar.add(newSecondMoment.sqrt())))
                             .add(value);

        value.assign(newValue);
      }
    });
  }

  dispose() {
    this.epsScalar.dispose();
    this.c.dispose();
    this.beta1Scalar.dispose();
    this.beta2Scalar.dispose();
    this.oneMinusBeta1.dispose();
    this.oneMinusBeta2.dispose();
    if (this.prevFirstMoment != null) {
      Object.keys(this.prevFirstMoment)
          .forEach(name => this.prevFirstMoment[name].dispose());
    }
    if (this.prevSecondMoment != null) {
      Object.keys(this.prevSecondMoment)
          .forEach(name => this.prevSecondMoment[name].dispose());
    }
  }
  getConfig(): ConfigDict {
    return {
      learningRate: this.learningRate,
      beta1: this.beta1,
      beta2: this.beta2,
      epsilon: this.epsilon,
    };
  }
  static fromConfig<T extends Serializable>(
      cls: SerializableConstructor<T>, config: ConfigDict): T {
    return new cls(
        config.learningRate, config.beta1, config.beta2, config.epsilon);
  }
}
SerializationMap.register(AMSgradOptimizer);
