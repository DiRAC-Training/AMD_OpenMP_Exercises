program main
  use omp_lib
  implicit none

#ifndef NO_UNIFIED_SHARED_MEMORY
!$omp requires unified_shared_memory
#endif

  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: NTIMERS = 1
  integer :: n, i, iter
  integer :: num_iteration
  real(dp) :: a, main_timer, main_start
  real(dp), allocatable :: x(:), y(:), z(:)
  real(dp), allocatable :: timers(:)
  real(dp) :: start, sum_time, min_time, max_time, avg_time
  character(len=32) :: arg

  n = 100000
  num_iteration = NTIMERS
  main_start = omp_get_wtime()

  if (command_argument_count() >= 1) then
     call get_command_argument(1, arg)
  end if

  a = 3.0_dp
  allocate(x(n), y(n), z(n))
!$omp target enter data map(alloc: x(1:n), y(1:n), z(1:n))

!$omp target teams distribute parallel do num_threads(64)
  do i = 1, n
     x(i) = 2.0_dp
     y(i) = 1.0_dp
  end do

  allocate(timers(num_iteration))
  timers = 0.0_dp

  do iter = 1, num_iteration
     start = omp_get_wtime()
     call daxpy(n, a, x, y, z)
     timers(iter) = omp_get_wtime() - start
  end do

  sum_time = 0.0_dp
  max_time = -1.0d10
  min_time = 1.0d10
  do iter = 1, num_iteration
     sum_time = sum_time + timers(iter)
     if (timers(iter) > max_time) max_time = timers(iter)
     if (timers(iter) < min_time) min_time = timers(iter)
  end do

  avg_time = sum_time / real(num_iteration, dp)

  write(*,'(A,F8.6,A,F8.6,A,F8.6)') '-Timing in Seconds: min=', min_time, ', max=', max_time, ', avg=', avg_time

  main_timer = omp_get_wtime() - main_start
  write(*,'(A,F8.6)') '-Overall time is ', main_timer

!$omp target update from(z(1))
  print '(A,I0,A,F0.6)', 'Last Value: z(', n, ')=', z(n)

!$omp target exit data map(delete: x(1:n), y(1:n), z(1:n))
  deallocate(x, y, z, timers)

end program main

subroutine daxpy(n, a, x, y, z)
  use omp_lib
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: a
  real(8), intent(in) :: x(n), y(n)
  real(8), intent(out) :: z(n)
  integer :: i

!$omp target teams distribute parallel do num_threads(64)
  do i = 1, n
     z(i) = a * x(i) + y(i)
  end do
end subroutine daxpy
