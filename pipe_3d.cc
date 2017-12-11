#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_accessor.templates.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include <map>
#include <sys/stat.h>

namespace TRL
{
using namespace dealii;

template <int dim>
class Heat_Pipe
{
public:
  Heat_Pipe(int argc, char *argv[]);
  ~Heat_Pipe();
  void run();

private:
  void read_mesh();
  void mesh_info(DoFHandler<dim> &dof_handler,
                 FE_Q<dim> &fe);
  void setup_system(DoFHandler<dim> &dof_handler,
                    FE_Q<dim> &fe,
                    parallel::shared::Triangulation<dim> &triangulation,
                    PETScWrappers::MPI::Vector &system_rhs,
                    PETScWrappers::MPI::Vector &solution,
                    PETScWrappers::MPI::Vector &old_solution,
                    PETScWrappers::MPI::SparseMatrix &system_matrix_3d,
                    PETScWrappers::MPI::SparseMatrix &mass_matrix_3d,
                    PETScWrappers::MPI::SparseMatrix &laplace_matrix_new_3d,
                    PETScWrappers::MPI::SparseMatrix &laplace_matrix_old_3d);
  void assemble_matrices();
  void assemble_matrices_1d();
  void assemble_system();
  void assemble_system_1d();
  unsigned int solve(PETScWrappers::MPI::Vector &system_rhs,
                     PETScWrappers::MPI::Vector &solution,
                     PETScWrappers::MPI::Vector &old_solution,
                     PETScWrappers::MPI::SparseMatrix &system_matrix);
  void output_results(DoFHandler<dim> &dof_handler,
                      parallel::shared::Triangulation<dim> &triangulation,
                      PETScWrappers::MPI::Vector &solution,
                      std::string name,
                      std::vector<std::pair<double, std::string>> &times_and_names);
  void initial_condition(DoFHandler<dim> &dof_handler,
                         PETScWrappers::MPI::Vector &solution,
                         PETScWrappers::MPI::Vector &old_solution);
  void assemble_system_transport_1d();

  ConstraintMatrix constraints;

  PETScWrappers::MPI::SparseMatrix system_matrix_3d;
  PETScWrappers::MPI::SparseMatrix mass_matrix_3d;
  PETScWrappers::MPI::SparseMatrix laplace_matrix_new_3d;
  PETScWrappers::MPI::SparseMatrix laplace_matrix_old_3d;
  PETScWrappers::MPI::Vector system_rhs_3d;
  PETScWrappers::MPI::Vector solution_3d, old_solution_3d;

  PETScWrappers::MPI::SparseMatrix system_matrix_1d;
  PETScWrappers::MPI::SparseMatrix mass_matrix_new_1d;
  PETScWrappers::MPI::SparseMatrix mass_matrix_old_1d;
  PETScWrappers::MPI::SparseMatrix laplace_matrix_new_1d;
  PETScWrappers::MPI::SparseMatrix laplace_matrix_old_1d;
  PETScWrappers::MPI::Vector system_rhs_1d;
  PETScWrappers::MPI::Vector solution_1d, old_solution_1d;

  parallel::shared::Triangulation<dim> triangulation_3d;
  parallel::shared::Triangulation<dim> triangulation_1d;

  FE_Q<dim> fe_3d;
  FE_Q<dim> fe_1d;
  DoFHandler<dim> dof_handler_3d;
  DoFHandler<dim> dof_handler_1d;

  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  ConditionalOStream pcout;

  //  IndexSet locally_owned_dofs;
  //  IndexSet locally_relevant_dofs;
  unsigned int n_local_cells;
  std::vector<types::global_dof_index> local_dofs_per_process;

  std::string mesh_path_3d;
  std::string mesh_path_1d;
  double time_step;
  double theta;
  double theta_1d;
  double time;
  unsigned int timestep_number;
  unsigned int timestep_number_max;
  std::vector<std::pair<double, std::string>> times_and_names_3d;
  std::vector<std::pair<double, std::string>> times_and_names_1d;
};

template <int dim>
Heat_Pipe<dim>::Heat_Pipe(int argc, char *argv[])
    : triangulation_3d(MPI_COMM_WORLD),
      triangulation_1d(MPI_COMM_WORLD),
      fe_3d(1),
      fe_1d(1),
      dof_handler_3d(triangulation_3d),
      dof_handler_1d(triangulation_1d),
      mpi_communicator(MPI_COMM_WORLD),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
      this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
      pcout(std::cout, (this_mpi_process == 0))
{
  mesh_path_3d = "/mnt/c/Users/Jose/Desktop/gmsh-3.0.6-Windows64/pipe_system_3d.msh";
  mesh_path_1d = "/mnt/c/Users/Jose/Desktop/gmsh-3.0.6-Windows64/mesh_1d.geo.msh";

  time_step = 10.;
  theta = 0.5;
  theta_1d = 0.5;
  time = 0;
  timestep_number = 0;
  timestep_number_max = 40;
}

template <int dim>
Heat_Pipe<dim>::~Heat_Pipe()
{
  dof_handler_3d.clear();
  dof_handler_1d.clear();
}

template <int dim>
void Heat_Pipe<dim>::read_mesh()
{
  {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation_3d);
    std::ifstream file(mesh_path_3d.c_str());
    grid_in.read_msh(file);
  }
  {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation_1d);
    std::ifstream file(mesh_path_1d.c_str());
    grid_in.read_msh(file);
  }
}

template <int dim>
void Heat_Pipe<dim>::mesh_info(DoFHandler<dim> &dof_handler,
                               FE_Q<dim> &fe)
{
  const QGauss<dim - 1> face_quadrature_formula(3);
  const unsigned int n_face_q_points = face_quadrature_formula.size();
  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                   update_values | update_gradients |
                                       update_quadrature_points | update_JxW_values);

  int number_of_total_local_cells = 0;
  int number_of_boundary_local_cells = 0;
  std::map<unsigned int, unsigned int> boundary_count;
  std::map<unsigned int, double> boundary_area;

  std::map<int, int> boundary_local_cells;
  std::map<int, int> general_local_cells;

  for (int i = 0; i < n_mpi_processes; i++)
  {
    general_local_cells[i] = 0;
    boundary_local_cells[i] = 0;
  }

  typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
  for (; cell != endc; ++cell)
  {
    number_of_total_local_cells++;
    if (cell->subdomain_id() == this_mpi_process)
      general_local_cells[this_mpi_process]++;
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      if (cell->face(face)->at_boundary())
      {
        fe_face_values.reinit(cell, face);
        if (cell->subdomain_id() == this_mpi_process)
        {
          number_of_boundary_local_cells++;
          boundary_local_cells[this_mpi_process]++;
        }

        double cell_area = 0.;
        for (unsigned int q_face_point = 0; q_face_point < n_face_q_points; ++q_face_point)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            cell_area +=
                fe_face_values.shape_value(i, q_face_point) *
                fe_face_values.JxW(q_face_point);
          }
        }

        unsigned int boundary_id = cell->face(face)->boundary_id();
        if (boundary_count.find(boundary_id) != boundary_count.end()) //boundary_id is in map
        {
          boundary_count[boundary_id]++;
          boundary_area[boundary_id] += cell_area;
        }
        else //boundary_id not in map
        {
          boundary_count[boundary_id] = 1;
          boundary_area[boundary_id] = cell_area;
        }
      }
  }

  pcout << "\n\tNumber of mpi processes: "
        << n_mpi_processes
        << std::endl;
  pcout << "\t------Mesh info------" << std::endl
        << "\t dimension..........: " << dim << std::endl
        << "\t total no. of total cells.: " << number_of_total_local_cells << std::endl;
  for (int i = 0; i < n_mpi_processes; i++)
  {
    pcout << "\t\tmpi " << i << ": "
          << "\t\t" << Utilities::MPI::sum(general_local_cells[i], mpi_communicator) << "\n";
  }

  pcout << "\n\t----Boundary info----" << std::endl
        << "\t total no. of boundary cells: "
        << Utilities::MPI::sum(number_of_boundary_local_cells, mpi_communicator) << std::endl;
  for (int i = 0; i < n_mpi_processes; i++)
  {
    pcout << "\t\tmpi " << i << ": "
          << "\t\t" << Utilities::MPI::sum(boundary_local_cells[i], mpi_communicator) << "\n";
  }
  pcout << "\n\t----Boundary indicators---- " << std::endl
        << "\tid\tcells\tarea(u^2)\n";
  for (std::map<unsigned int, unsigned int>::iterator
           it = boundary_count.begin();
       it != boundary_count.end(); ++it)
    pcout << "\t" << it->first
          << "\t" << it->second
          << "\t" << boundary_area[it->first]
          << std::endl;
  pcout << std::endl
        << std::endl;
}

template <int dim>
void Heat_Pipe<dim>::setup_system(DoFHandler<dim> &dof_handler,
                                  FE_Q<dim> &fe,
                                  parallel::shared::Triangulation<dim> &triangulation,
                                  PETScWrappers::MPI::Vector &system_rhs,
                                  PETScWrappers::MPI::Vector &solution,
                                  PETScWrappers::MPI::Vector &old_solution,
                                  PETScWrappers::MPI::SparseMatrix &system_matrix,
                                  PETScWrappers::MPI::SparseMatrix &mass_matrix,
                                  PETScWrappers::MPI::SparseMatrix &laplace_matrix_new,
                                  PETScWrappers::MPI::SparseMatrix &laplace_matrix_old)
{
  dof_handler.distribute_dofs(fe);
  IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  n_local_cells =
      GridTools::count_cells_with_subdomain_association(triangulation,
                                                        triangulation.locally_owned_subdomain());

  local_dofs_per_process = dof_handler.n_locally_owned_dofs_per_processor();

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern, constraints, /*keep constrained dofs*/ false);
  SparsityTools::distribute_sparsity_pattern(sparsity_pattern, local_dofs_per_process,
                                             mpi_communicator, locally_relevant_dofs);

  old_solution
      .reinit(locally_owned_dofs, mpi_communicator);
  solution
      .reinit(locally_owned_dofs, mpi_communicator);
  system_rhs
      .reinit(locally_owned_dofs, mpi_communicator);

  system_matrix
      .reinit(locally_owned_dofs,
              locally_owned_dofs,
              sparsity_pattern,
              mpi_communicator);
  mass_matrix
      .reinit(locally_owned_dofs,
              locally_owned_dofs,
              sparsity_pattern,
              mpi_communicator);
  laplace_matrix_new
      .reinit(locally_owned_dofs,
              locally_owned_dofs,
              sparsity_pattern,
              mpi_communicator);
  laplace_matrix_old
      .reinit(locally_owned_dofs,
              locally_owned_dofs,
              sparsity_pattern,
              mpi_communicator);
}

template <int dim>
void Heat_Pipe<dim>::assemble_matrices()
{
  mass_matrix_3d = 0.;
  laplace_matrix_new_3d = 0.;
  laplace_matrix_old_3d = 0.;
  system_rhs_3d = 0.;
  mass_matrix_3d.compress(VectorOperation::insert);
  laplace_matrix_new_3d.compress(VectorOperation::insert);
  laplace_matrix_old_3d.compress(VectorOperation::insert);
  system_rhs_3d.compress(VectorOperation::insert);

  const QGauss<dim> quadrature_formula(3);
  const QGauss<dim - 1> face_quadrature_formula(3);
  FEValues<dim> fe_values(fe_3d, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values(fe_3d, face_quadrature_formula,
                                   update_values | update_gradients |
                                       update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe_3d.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_laplace_matrix_new(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_laplace_matrix_old(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<unsigned int> local_dof_indices(dofs_per_cell);
  // const Vector<double> localized_old_solution(old_solution_3d);
  // const Vector<double> localized_new_solution(solution_3d);

  double face_boundary_indicator;
  unsigned int faces_on_road_surface = 0;
  unsigned int faces_on_soil_surface = 0;

  typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_3d.begin_active(),
      endc = dof_handler_3d.end();
  for (; cell != endc; ++cell)
    if (cell->subdomain_id() == this_mpi_process)
    {
      fe_values.reinit(cell);
      cell_mass_matrix = 0.;
      cell_laplace_matrix_new = 0.;
      cell_laplace_matrix_old = 0.;
      cell_rhs = 0.;

      double thermal_conductivity = 1.;
      double thermal_heat_capacity = 1000.;
      double density = 1000.;

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            cell_mass_matrix(i, j) +=
                thermal_heat_capacity * density *
                fe_values.shape_value(i, q_point) *
                fe_values.shape_value(j, q_point) *
                fe_values.JxW(q_point);
            cell_laplace_matrix_new(i, j) +=
                thermal_conductivity *
                fe_values.shape_grad(i, q_point) *
                fe_values.shape_grad(j, q_point) *
                fe_values.JxW(q_point);
            cell_laplace_matrix_old(i, j) +=
                thermal_conductivity *
                fe_values.shape_grad(i, q_point) *
                fe_values.shape_grad(j, q_point) *
                fe_values.JxW(q_point);
          }

      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      {
        face_boundary_indicator = cell->face(face)->boundary_id();
        if (cell->face(face)->at_boundary() &&
            ((face_boundary_indicator == 5) ||
             (face_boundary_indicator == 6)))
        {
          /*
          * Based on boundary_id, build a function to return:
          * -- Flow 
          * -- Convective coefficient
          * -- Boundary codition type (2nd or 3rd)
          */
          double outbound_convective_coefficient_new = 0.;
          double outbound_convective_coefficient_old = 0.;
          double inbound_heat_flux_new = 0.;
          double inbound_heat_flux_old = 0.;
          std::string boundary_condition_type = "";

          if (face_boundary_indicator == 5)
          {
            outbound_convective_coefficient_old = 1.;
            inbound_heat_flux_old = 100.;
            outbound_convective_coefficient_new = 1.;
            inbound_heat_flux_new = 100.;
            boundary_condition_type = "2nd";
          }
          else if (face_boundary_indicator == 6)
          {
            outbound_convective_coefficient_old = 1.;
            inbound_heat_flux_old = 100.;
            outbound_convective_coefficient_new = 1.;
            inbound_heat_flux_new = 100.;
            boundary_condition_type = "2nd";
          }
          else
          {
            pcout << "Error: author not implemented." << std::endl
                  << "Error in assembling function." << std::endl;
            throw 3;
          }

          fe_face_values.reinit(cell, face);
          if (boundary_condition_type.compare("2nd") == 0 ||
              boundary_condition_type.compare("3rd") == 0)
          {
            for (unsigned int q_face_point = 0; q_face_point < n_face_q_points; ++q_face_point)
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                if (boundary_condition_type.compare("3rd") == 0)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    cell_laplace_matrix_new(i, j) +=
                        outbound_convective_coefficient_new *
                        fe_face_values.shape_value(i, q_face_point) *
                        fe_face_values.shape_value(j, q_face_point) *
                        fe_face_values.JxW(q_face_point);
                    cell_laplace_matrix_old(i, j) +=
                        outbound_convective_coefficient_old *
                        fe_face_values.shape_value(i, q_face_point) *
                        fe_face_values.shape_value(j, q_face_point) *
                        fe_face_values.JxW(q_face_point);
                  }
                }
                cell_rhs(i) +=
                    inbound_heat_flux_new *
                        time_step * theta *
                        fe_face_values.shape_value(i, q_face_point) *
                        fe_face_values.JxW(q_face_point) +
                    inbound_heat_flux_old *
                        time_step * (1. - theta) *
                        fe_face_values.shape_value(i, q_face_point) *
                        fe_face_values.JxW(q_face_point);
              }
          }
        }
      }
      cell->get_dof_indices(local_dof_indices);
      constraints
          .distribute_local_to_global(cell_laplace_matrix_new,
                                      local_dof_indices,
                                      laplace_matrix_new_3d);
      constraints
          .distribute_local_to_global(cell_laplace_matrix_old,
                                      local_dof_indices,
                                      laplace_matrix_old_3d);
      constraints
          .distribute_local_to_global(cell_mass_matrix,
                                      cell_rhs,
                                      local_dof_indices,
                                      mass_matrix_3d, system_rhs_3d);
    }
  laplace_matrix_new_3d.compress(VectorOperation::add);
  laplace_matrix_old_3d.compress(VectorOperation::add);
  mass_matrix_3d.compress(VectorOperation::add);
  system_rhs_3d.compress(VectorOperation::add);
}

template <int dim>
void Heat_Pipe<dim>::assemble_system()
{
  PETScWrappers::MPI::Vector tmp;
  tmp.reinit(solution_3d);

  laplace_matrix_old_3d.vmult(tmp, old_solution_3d);
  tmp.compress(VectorOperation::insert);
  tmp *= -(1. - theta) * time_step;
  tmp.compress(VectorOperation::insert);

  system_rhs_3d += tmp;
  system_rhs_3d.compress(VectorOperation::add);

  mass_matrix_3d.vmult(tmp, old_solution_3d);
  tmp.compress(VectorOperation::insert);

  system_rhs_3d += tmp;
  system_rhs_3d.compress(VectorOperation::add);

  system_matrix_3d = 0.;
  system_matrix_3d.compress(VectorOperation::insert);
  system_matrix_3d.add(1.0, mass_matrix_3d);
  system_matrix_3d.compress(VectorOperation::add);

  system_matrix_3d.add(theta * time_step, laplace_matrix_new_3d);
  system_matrix_3d.compress(VectorOperation::add);

  std::map<unsigned int, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler_3d,
                                           900,
                                           ConstantFunction<dim>(0.),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix_3d,
                                     solution_3d,
                                     system_rhs_3d,
                                     false);
}

template <int dim>
unsigned int Heat_Pipe<dim>::solve(PETScWrappers::MPI::Vector &system_rhs,
                                   PETScWrappers::MPI::Vector &solution,
                                   PETScWrappers::MPI::Vector &old_solution,
                                   PETScWrappers::MPI::SparseMatrix &system_matrix)
{
  SolverControl solver_control(100 * solution.size(),
                               1e-8 * system_rhs.l2_norm());
  // PETScWrappers::SolverCG solver(solver_control,
  //                                mpi_communicator);
  PETScWrappers::SolverBicgstab solver(solver_control,
                                       mpi_communicator);

  PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
  solver.solve(system_matrix, solution,
               system_rhs, preconditioner);
  Vector<double> localized_solution(solution);
  constraints.distribute(localized_solution);
  solution = localized_solution;
  solution.compress(VectorOperation::insert);
  return solver_control.last_step();

  // SolverControl solver_control(100 * solution.size(),
  //                              1e-8 * system_rhs.l2_norm());
  // SolverBicgstab<> bicgstab(solver_control);
  // PreconditionJacobi<> preconditioner;
  // preconditioner
  //     .initialize(system_matrix, 1.0);
  // bicgstab
  //     .solve(system_matrix, solution,
  //            system_rhs, preconditioner);
  // hanging_node_constraints.distribute(solution);
}

template <int dim>
void Heat_Pipe<dim>::output_results(DoFHandler<dim> &dof_handler,
                                    parallel::shared::Triangulation<dim> &triangulation,
                                    PETScWrappers::MPI::Vector &solution,
                                    std::string name,
                                    std::vector<std::pair<double, std::string>> &times_and_names)
{
  const Vector<double> localized_solution(solution);
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(localized_solution, "temperature");
  /*
  * Add information about in which mpi process is each cell being processed
  */
  std::vector<unsigned int> partition_int(triangulation.n_active_cells());
  GridTools::get_subdomain_association(triangulation, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");
  /*
  * Define output name
  */
  std::string filename = name + "-" + Utilities::int_to_string(timestep_number, 4) + "." + Utilities::int_to_string(this_mpi_process, 3) + ".vtu";

  data_out.build_patches();
  std::ofstream output(filename.c_str());
  data_out.write_vtu(output);

  if (this_mpi_process == 0)
  {
    std::vector<std::string> filenames;
    for (unsigned int i = 0; i < n_mpi_processes; ++i)
      filenames.push_back(name + "-" + Utilities::int_to_string(timestep_number, 4) + "." + Utilities::int_to_string(i, 3) + ".vtu");

    const std::string
        pvtu_master_filename = (name + "-" + Utilities::int_to_string(timestep_number, 4) + ".pvtu");
    std::ofstream pvtu_master(pvtu_master_filename.c_str());
    data_out.write_pvtu_record(pvtu_master, filenames);

    //    static std::vector<std::pair<double, std::string>> times_and_names;
    times_and_names.push_back(std::pair<double, std::string>(time - time_step, pvtu_master_filename));
    std::ofstream pvd_output(name + ".pvd");
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
  }
}

template <int dim>
void Heat_Pipe<dim>::initial_condition(DoFHandler<dim> &dof_handler,
                                       PETScWrappers::MPI::Vector &solution,
                                       PETScWrappers::MPI::Vector &old_solution)
{
  VectorTools::project(dof_handler,
                       constraints, QGauss<dim>(3),
                       ConstantFunction<dim>(0.),
                       old_solution);
  old_solution.compress(VectorOperation::insert);
  solution = old_solution;
  solution.compress(VectorOperation::insert);
}

template <int dim>
void Heat_Pipe<dim>::assemble_matrices_1d()
{
  mass_matrix_new_1d = 0.;
  mass_matrix_old_1d = 0.;
  laplace_matrix_new_1d = 0.;
  laplace_matrix_old_1d = 0.;
  system_rhs_1d = 0.;
  system_matrix_1d = 0.;

  mass_matrix_new_1d.compress(VectorOperation::insert);
  mass_matrix_old_1d.compress(VectorOperation::insert);
  laplace_matrix_new_1d.compress(VectorOperation::insert);
  laplace_matrix_old_1d.compress(VectorOperation::insert);
  system_rhs_1d.compress(VectorOperation::insert);
  system_matrix_1d.compress(VectorOperation::insert);

  QGauss<dim> quadrature_formula(2);
  QGauss<dim - 1> face_quadrature_formula(2);
  FEValues<dim> fe_values(fe_1d, quadrature_formula,
                          update_values | update_gradients | update_hessians |
                              update_JxW_values);
  FEFaceValues<dim> fe_face_values(fe_1d, face_quadrature_formula,
                                   update_values | update_gradients |
                                       update_normal_vectors |
                                       update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe_1d.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  FullMatrix<double> cell_mass_matrix_new(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix_old(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_laplace_matrix_new(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_laplace_matrix_old(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<unsigned int> local_dof_indices(fe_3d.dofs_per_cell);
  double face_boundary_indicator;
  unsigned int cell_index = 0;

  typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_1d.begin_active(),
      endc = dof_handler_1d.end();
  for (; cell != endc; ++cell)
  {
    fe_values.reinit(cell);
    cell_mass_matrix_new = 0;
    cell_mass_matrix_old = 0;
    cell_laplace_matrix_new = 0;
    cell_laplace_matrix_old = 0;
    cell_rhs = 0;

    double initializer[dim];
    for (int i = 0; i < dim; i++)
      initializer[i] = 0;
    Tensor<1, dim> new_velocity(initializer);
    Tensor<1, dim> old_velocity(initializer);

    new_velocity[dim - 1] = 0.42E-0;//m/s
    old_velocity[dim - 1] = 0.42E-0;//m/s

    double new_free_moisture_content = 1.;
    double old_free_moisture_content = 1.;

    double new_diffusion_value = new_velocity.norm()+1E-5;
    double old_diffusion_value = old_velocity.norm()+1E-5;

    double beta = 1.;
    double tau = 0.;
    double Peclet = 0.;
    if (new_diffusion_value > 1.E-12)
    {
      Peclet =
          0.5 * cell->diameter() * (0.5 * new_velocity.norm() + 0.5 * old_velocity.norm()) /
          (0.5 * new_diffusion_value + 0.5 * old_diffusion_value);
    }
    if (Peclet > 1.E-12)
    {
      beta = 1. / tanh(Peclet) - 1. / Peclet;
    }
    if (new_velocity.norm() > 0.)
    {
      tau = 0.5 * beta * cell->diameter() / (0.5 * new_velocity.norm() + 0.5 * old_velocity.norm());
    }

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        double new_sink_factor = 0;
        double old_sink_factor = 0;

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            /*i=test function, j=concentration IMPORTANT!!*/
            cell_mass_matrix_new(i, j) +=
                (fe_values.shape_value(i, q_point) +
                 tau *
                     new_velocity *
                     fe_values.shape_grad(i, q_point)) *
                fe_values.shape_value(j, q_point) *
                new_free_moisture_content *
                fe_values.shape_value(k, q_point) *
                fe_values.JxW(q_point);

            cell_mass_matrix_old(i, j) +=
                (fe_values.shape_value(i, q_point) +
                 tau *
                     old_velocity *
                     fe_values.shape_grad(i, q_point)) *
                fe_values.shape_value(j, q_point) *
                old_free_moisture_content *
                fe_values.shape_value(k, q_point) *
                fe_values.JxW(q_point);

            cell_laplace_matrix_new(i, j) +=
                /*Diffusive term*/
                fe_values.shape_grad(i, q_point) *
                    fe_values.shape_grad(j, q_point) *
                    new_diffusion_value *
                    new_free_moisture_content *
                    fe_values.shape_value(k, q_point) *
                    fe_values.JxW(q_point) +
                /*Convective term*/
                (
                    fe_values.shape_value(i, q_point) +
                    tau *
                        new_velocity *
                        fe_values.shape_grad(i, q_point)) *
                    fe_values.shape_grad(j, q_point) *
                    new_velocity *
                    fe_values.shape_value(k, q_point) *
                    fe_values.JxW(q_point)
                /*Reaction term*/
                -
                (fe_values.shape_value(i, q_point) +
                 tau *
                     new_velocity *
                     fe_values.shape_grad(i, q_point)) *
                    fe_values.shape_value(j, q_point) *
                    new_sink_factor *
                    fe_values.shape_value(k, q_point) *
                    fe_values.JxW(q_point);

            cell_laplace_matrix_old(i, j) +=
                /*Diffusive term*/
                fe_values.shape_grad(i, q_point) *
                    fe_values.shape_grad(j, q_point) *
                    old_diffusion_value *
                    old_free_moisture_content *
                    fe_values.shape_value(k, q_point) *
                    fe_values.JxW(q_point) +
                /*Convective term*/
                (
                    fe_values.shape_value(i, q_point) +
                    tau *
                        old_velocity *
                        fe_values.shape_grad(i, q_point)) *
                    fe_values.shape_grad(j, q_point) *
                    old_velocity *
                    fe_values.shape_value(k, q_point) *
                    fe_values.JxW(q_point)
                /*Reaction term*/
                -
                (fe_values.shape_value(i, q_point) +
                 tau *
                     old_velocity *
                     fe_values.shape_grad(i, q_point)) *
                    fe_values.shape_value(j, q_point) *
                    old_sink_factor *
                    fe_values.shape_value(k, q_point) *
                    fe_values.JxW(q_point);
          }
        }
      }
    }

    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
    {
      if (cell->face(face)->at_boundary())
      {
        fe_face_values.reinit(cell, face);
        face_boundary_indicator = cell->face(face)->boundary_id();
        if (face_boundary_indicator == 50)
        {
          for (unsigned int q_face_point = 0; q_face_point < n_face_q_points; ++q_face_point)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              double concentration_at_boundary = 10.; //mg_substrate/cm3_total_water

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                { /*i=test function, j=concentration IMPORTANT!!*/
                  cell_laplace_matrix_new(i, j) -=
                      (fe_face_values.shape_value(i, q_face_point) +
                       tau *
                           new_velocity *
                           fe_face_values.shape_grad(i, q_face_point)) *
                      fe_face_values.shape_value(j, q_face_point) *
                      new_velocity *
                      fe_face_values.normal_vector(q_face_point) *
                      fe_face_values.shape_value(k, q_face_point) *
                      fe_face_values.JxW(q_face_point);

                  cell_laplace_matrix_old(i, j) -=
                      (fe_face_values.shape_value(i, q_face_point) +
                       tau *
                           old_velocity *
                           fe_face_values.shape_grad(i, q_face_point)) *
                      fe_face_values.shape_value(j, q_face_point) *
                      old_velocity *
                      fe_face_values.normal_vector(q_face_point) *
                      fe_face_values.shape_value(k, q_face_point) *
                      fe_face_values.JxW(q_face_point);
                }
                cell_rhs(i) -=
                    (fe_face_values.shape_value(i, q_face_point) +
                     tau *
                         new_velocity *
                         fe_face_values.shape_grad(i, q_face_point)) *
                        time_step *
                        (theta_1d)*concentration_at_boundary *
                        new_velocity *
                        fe_face_values.normal_vector(q_face_point) *
                        fe_face_values.shape_value(k, q_face_point) *
                        fe_face_values.JxW(q_face_point) +
                    (fe_face_values.shape_value(i, q_face_point) +
                     tau *
                         new_velocity *
                         fe_face_values.shape_grad(i, q_face_point)) *
                        time_step *
                        (1. - theta_1d) *
                        concentration_at_boundary *
                        old_velocity *
                        fe_face_values.normal_vector(q_face_point) *
                        fe_face_values.shape_value(k, q_face_point) *
                        fe_face_values.JxW(q_face_point);
              }
            }
          }
        }
      }
    }
    cell->get_dof_indices(local_dof_indices);
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        laplace_matrix_new_1d
            .add(local_dof_indices[i], local_dof_indices[j], cell_laplace_matrix_new(i, j));
        laplace_matrix_old_1d
            .add(local_dof_indices[i], local_dof_indices[j], cell_laplace_matrix_old(i, j));
        mass_matrix_new_1d
            .add(local_dof_indices[i], local_dof_indices[j], cell_mass_matrix_new(i, j));
        mass_matrix_old_1d
            .add(local_dof_indices[i], local_dof_indices[j], cell_mass_matrix_old(i, j));
      }
      system_rhs_1d(local_dof_indices[i]) += cell_rhs(i);
    }
  }
  laplace_matrix_new_1d.compress(VectorOperation::add);
  laplace_matrix_old_1d.compress(VectorOperation::add);
  mass_matrix_new_1d.compress(VectorOperation::add);
  mass_matrix_old_1d.compress(VectorOperation::add);
  system_rhs_1d.compress(VectorOperation::add);
}

template <int dim>
void Heat_Pipe<dim>::assemble_system_1d()
{
  PETScWrappers::MPI::Vector tmp;
  tmp.reinit(solution_1d);

  laplace_matrix_old_1d.vmult(tmp, old_solution_1d);
  tmp.compress(VectorOperation::insert);
  tmp *= -(1. - theta) * time_step;
  tmp.compress(VectorOperation::insert);

  system_rhs_1d += tmp;
  system_rhs_1d.compress(VectorOperation::add);

  mass_matrix_old_1d.vmult(tmp, old_solution_1d);
  tmp.compress(VectorOperation::insert);

  system_rhs_1d += tmp;
  system_rhs_1d.compress(VectorOperation::add);

  system_matrix_1d = 0.;
  system_matrix_1d.compress(VectorOperation::insert);
  system_matrix_1d.add(1.0, mass_matrix_new_1d);
  system_matrix_1d.compress(VectorOperation::add);

  system_matrix_1d.add(theta * time_step, laplace_matrix_new_1d);
  system_matrix_1d.compress(VectorOperation::add);

  std::map<unsigned int, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler_1d,
                                           900,
                                           ConstantFunction<dim>(0.),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix_1d,
                                     solution_1d,
                                     system_rhs_1d,
                                     false);
}

template <int dim>
void Heat_Pipe<dim>::run()
{
  TimerOutput timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::cpu_and_wall_times);
  {
    TimerOutput::Scope timer_section(timer, "Read mesh");
    read_mesh();
  }
  {
    TimerOutput::Scope timer_section(timer, "Setup system");
    setup_system(dof_handler_3d,
                 fe_3d,
                 triangulation_3d,
                 system_rhs_3d,
                 solution_3d,
                 old_solution_3d,
                 system_matrix_3d,
                 mass_matrix_3d,
                 laplace_matrix_new_3d,
                 laplace_matrix_old_3d);

    setup_system(dof_handler_1d,
                 fe_1d,
                 triangulation_1d,
                 system_rhs_1d,
                 solution_1d,
                 old_solution_1d,
                 system_matrix_1d,
                 mass_matrix_old_1d,
                 laplace_matrix_new_1d,
                 laplace_matrix_old_1d);
    mass_matrix_new_1d.reinit(mass_matrix_old_1d);
  }
  {
    TimerOutput::Scope timer_section(timer, "Set initial condition");
    initial_condition(dof_handler_3d,
                      solution_3d,
                      old_solution_3d);
    initial_condition(dof_handler_1d,
                      solution_1d,
                      old_solution_1d);
  }
  {
    TimerOutput::Scope timer_section(timer, "Mesh info");
    mesh_info(dof_handler_3d, fe_3d);
    mesh_info(dof_handler_1d, fe_1d);
  }

  for (timestep_number = 1, time = time_step;
       timestep_number <= timestep_number_max;
       timestep_number++, time += time_step)
  {
    {
      TimerOutput::Scope timer_section(timer, "Assemble matrices");
      //assemble_matrices();
      assemble_matrices_1d();
    }
    {
      TimerOutput::Scope timer_section(timer, "Assemble system");
      //assemble_system();
      assemble_system_1d();
    }
    {
      TimerOutput::Scope timer_section(timer, "Solve");
      // solve(system_rhs_3d,
      //       solution_3d,
      //       old_solution_3d,
      //       system_matrix_3d);
      solve(system_rhs_1d,
            solution_1d,
            old_solution_1d,
            system_matrix_1d);
    }
    {
      TimerOutput::Scope timer_section(timer, "Output results");
      //output_results(dof_handler_3d,
      //  triangulation_3d,
      //  solution_3d,
      //  "solution-3d",
      //  times_and_names_3d);
      output_results(dof_handler_1d,
                     triangulation_1d,
                     solution_1d,
                     "solution-1d",
                     times_and_names_1d);
    }

    pcout << "timestep: " << timestep_number << std::endl;

    //old_solution_3d = solution_3d;
    old_solution_1d = solution_1d;
  }
}
}

int main(int argc, char **argv)
{
  const unsigned int dim = 3;

  try
  {
    using namespace dealii;
    using namespace TRL;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    {
      deallog.depth_console(0);

      Heat_Pipe<dim> trl_problem(argc, argv);
      trl_problem.run();
    }
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
