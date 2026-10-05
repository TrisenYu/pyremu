// poisson.cc — DUNE 有限元求解二维 Poisson 方程 (数值偏微分)
//
// 定解问题 (Dirichlet 边界, 单位方域 [0,1]^2):
//     -∇²u = f,  u|_边界 = 0
// 取精确解 u(x,y) = sin(πx) sin(πy), 则右端 f = 2π² sin(πx) sin(πy)。
//
// 离散: YaspGrid 结构化 N×N 网格 + Q1 (双线性) Lagrange 基, 组装刚度阵与
// 载荷向量, 施加齐次 Dirichlet 边界后以 ISTL 共轭梯度 (CG) + SeqILDL 预条件求解。
//
// 自检: Q1 元在 L2 范数下二阶收敛, 网格加密一倍误差降为 1/4。用 N=8 与
// N=16 两套网格的节点均方根误差之比验证 (预期 ratio ≈ 4)。
//
// 组装/边界辅助函数取自 dune-functions 的 poisson-pq2 示例 (MIT/自用许可),
// 针对 Q1 与齐次边界做了简化。
//
// 与 Rust 载荷 fem 的关系: fem 原先求解本文件这一常系数 Poisson 问题, 后改为变系数
// 扩散反应方程 (P1 三角形单元 + 复系数)。两者现为不同的定解问题与不同的离散方式,
// 数值结果 (误差与收敛比) 不可互相核对, 仅各自与自身的精确解比较。

#include <cmath>
#include <cstdio>
#include <vector>

#include <dune/common/fvector.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/rangeutilities.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/grid/yaspgrid.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvers.hh>

#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/gridfunctions/gridviewfunction.hh>

using namespace Dune;

// 计算单个单元刚度阵 (参考元上积分, 经几何映射到真实单元)
template <class LocalView, class MatrixType>
void getLocalMatrix(const LocalView &localView, MatrixType &elementMatrix) {
	using Element		   = typename LocalView::Element;
	const Element &element = localView.element();
	const int dim		   = Element::dimension;
	auto geometry		   = element.geometry();

	const auto &localFiniteElement = localView.tree().finiteElement();

	elementMatrix.setSize(
		localFiniteElement.localBasis().size(), localFiniteElement.localBasis().size());
	elementMatrix = 0;

	int order = 2 * (dim * localFiniteElement.localBasis().order() - 1);
	const QuadratureRule<double, dim> &quad =
		QuadratureRules<double, dim>::rule(element.type(), order);

	for (size_t pt = 0; pt < quad.size(); pt++) {
		const FieldVector<double, dim> &quadPos = quad[pt].position();
		const auto &jacobianInverse				= geometry.jacobianInverse(quadPos);
		const double integrationElement			= geometry.integrationElement(quadPos);

		std::vector<FieldMatrix<double, 1, dim>> referenceJacobians;
		localFiniteElement.localBasis().evaluateJacobian(quadPos, referenceJacobians);

		std::vector<FieldMatrix<double, 1, dim>> jacobians(referenceJacobians.size());
		for (size_t i = 0; i < jacobians.size(); i++) {
			jacobians[i] = referenceJacobians[i] * jacobianInverse;
		}

		for (size_t i = 0; i < elementMatrix.N(); i++) {
			for (size_t j = 0; j < elementMatrix.M(); j++) {
				elementMatrix[i][j] += (jacobians[i] * transpose(jacobians[j]))
									   * quad[pt].weight() * integrationElement;
			}
		}
	}
}

// 计算单个单元的载荷向量
template <class LocalView, class LocalVolumeTerm>
void getVolumeTerm(
	const LocalView &localView,
	BlockVector<double> &localRhs,
	LocalVolumeTerm &&localVolumeTerm) {
	using Element		   = typename LocalView::Element;
	const Element &element = localView.element();
	const int dim		   = Element::dimension;

	const auto &localFiniteElement = localView.tree().finiteElement();

	localRhs.resize(localFiniteElement.localBasis().size());
	localRhs = 0;

	int order = dim * localFiniteElement.localBasis().order();
	const QuadratureRule<double, dim> &quad =
		QuadratureRules<double, dim>::rule(element.type(), order);

	for (size_t pt = 0; pt < quad.size(); pt++) {
		const FieldVector<double, dim> &quadPos = quad[pt].position();
		const double integrationElement = element.geometry().integrationElement(quadPos);
		double functionValue			= localVolumeTerm(quadPos);

		std::vector<FieldVector<double, 1>> shapeFunctionValues;
		localFiniteElement.localBasis().evaluateFunction(quadPos, shapeFunctionValues);

		for (size_t i = 0; i < localRhs.size(); i++) {
			localRhs[i] += shapeFunctionValues[i] * functionValue * quad[pt].weight()
						   * integrationElement;
		}
	}
}

// 生成刚度阵非零占位模式
template <class FEBasis>
void getOccupationPattern(const FEBasis &feBasis, MatrixIndexSet &nb) {
	auto n = feBasis.size();
	nb.resize(n, n);

	auto localView = feBasis.localView();

	for (const auto &e : elements(feBasis.gridView())) {
		localView.bind(e);
		for (size_t i = 0; i < localView.tree().size(); i++) {
			for (size_t j = 0; j < localView.tree().size(); j++) {
				nb.add(localView.index(i), localView.index(j));
			}
		}
	}
}

// 组装 Laplace 刚度阵与载荷向量
template <class FEBasis, class VolumeTerm>
void assembleLaplaceMatrix(
	const FEBasis &feBasis,
	BCRSMatrix<double> &matrix,
	BlockVector<double> &rhs,
	VolumeTerm &&volumeTerm) {
	using GridView	  = typename FEBasis::GridView;
	GridView gridView = feBasis.gridView();

	auto localVolumeTerm =
		localFunction(Functions::makeGridViewFunction(volumeTerm, gridView));

	MatrixIndexSet occupationPattern;
	getOccupationPattern(feBasis, occupationPattern);
	occupationPattern.exportIdx(matrix);

	rhs.resize(feBasis.size());
	matrix = 0;
	rhs	   = 0;

	auto localView = feBasis.localView();

	for (const auto &e : elements(gridView)) {
		localView.bind(e);

		Matrix<double> elementMatrix;
		getLocalMatrix(localView, elementMatrix);

		for (size_t i = 0; i < elementMatrix.N(); i++) {
			auto row = localView.index(i);
			for (size_t j = 0; j < elementMatrix.M(); j++) {
				matrix[row][localView.index(j)] += elementMatrix[i][j];
			}
		}

		BlockVector<double> localRhs;
		localVolumeTerm.bind(e);
		getVolumeTerm(localView, localRhs, localVolumeTerm);

		for (size_t i = 0; i < localRhs.size(); i++) {
			rhs[localView.index(i)] += localRhs[i];
		}
	}
}

// 标记位于网格边界上的 Lagrange 节点 (用作 Dirichlet 节点)
template <class FEBasis>
void boundaryTreatment(const FEBasis &feBasis, std::vector<char> &dirichletNodes) {
	dirichletNodes.clear();
	dirichletNodes.resize(feBasis.size(), false);

	Functions::forEachBoundaryDOF(
		feBasis, [&](auto &&index) { dirichletNodes[index] = true; });
}

// 在 N×N 网格上求解 Poisson 方程, 返回节点均方根误差 (相对精确解)
static double solvePoisson(int n) {
	const double pi = std::acos(-1.0);
	auto exact		= [pi](const auto &x) {
		 return std::sin(pi * x[0]) * std::sin(pi * x[1]);
	};
	auto rhs = [pi](const auto &x) {
		return 2.0 * pi * pi * std::sin(pi * x[0]) * std::sin(pi * x[1]);
	};

	Dune::YaspGrid<2> grid({1.0, 1.0}, {n, n});
	auto gridView  = grid.leafGridView();
	using GridView = decltype(gridView);

	using FEBasis = Functions::LagrangeBasis<GridView, 1>;
	FEBasis feBasis(gridView);

	using VectorType = BlockVector<double>;
	using MatrixType = BCRSMatrix<double>;

	VectorType x(feBasis.size());
	x = 0;

	VectorType b;
	MatrixType A;
	assembleLaplaceMatrix(feBasis, A, b, rhs);

	// 齐次 Dirichlet 边界 (u=0): 对角置 1、行置 0, 对称地抹去边界列贡献
	std::vector<char> dirichletNodes;
	boundaryTreatment(feBasis, dirichletNodes);
	for (size_t i = 0; i < A.N(); i++) {
		if (dirichletNodes[i]) {
			b[i] = 0.0;
			for (auto &&[entry, idx] : sparseRange(A[i])) {
				entry = (i == idx) ? 1.0 : 0.0;
			}
		} else {
			for (auto &&[entry, idx] : sparseRange(A[i])) {
				if (dirichletNodes[idx]) {
					entry = 0.0;
				}
			}
		}
	}

	MatrixAdapter<MatrixType, VectorType, VectorType> op(A);
	SeqILDL<MatrixType, VectorType, VectorType> ildl(A, 1.0);
	CGSolver<VectorType> cg(op, ildl, 1e-4, 100, 0);
	InverseOperatorResult statistics;
	cg.apply(x, b, statistics);

	// 节点均方根误差: 与精确解在 DOF 处逐点比较
	std::vector<double> exactAtDofs(feBasis.size());
	interpolate(feBasis, exactAtDofs, exact);
	double err2 = 0.0;
	for (size_t i = 0; i < feBasis.size(); i++) {
		double d = x[i] - exactAtDofs[i];
		err2 += d * d;
	}
	return std::sqrt(err2 / static_cast<double>(feBasis.size()));
}

int main(int argc, char **argv) {
	MPIHelper::instance(argc, argv);

	double e8	 = solvePoisson(8);
	double e16	 = solvePoisson(16);
	double ratio = e8 / e16;

	std::printf(
		"dune-poisson: err(N=8)=%.6e err(N=16)=%.6e ratio=%.3f\n", e8, e16, ratio);

	// Q1 二阶收敛: 网格加密一倍误差降为 1/4 (容忍一定离散与求解误差)
	bool ok = (ratio > 3.5) && (ratio < 4.5);
	std::printf("[regress] dune-poisson: %s\n", ok ? "PASS" : "FAIL");
	return ok ? 0 : 1;
}
