from typing import Generic, TypeVar

from hyped.common._generic import solve_typevar


class TestResolveTypeVar:
    def test_solve_typevar_unset(self) -> None:
        T = TypeVar("T")

        class A(Generic[T]):
            pass

        assert solve_typevar(A, T) is None

    def test_solve_typevar_fallback_to_bound(self) -> None:
        T = TypeVar("T", bound=int)

        class A(Generic[T]):
            pass

        assert solve_typevar(A, T) is int

    def test_solve_typevar_easy(self) -> None:
        T = TypeVar("T")

        class A:
            pass

        class B(Generic[T]):
            pass

        class C(B[A]):
            pass

        assert solve_typevar(C, T) == A

    def test_solve_inline_generic(self) -> None:
        T = TypeVar("T")

        class A:
            pass

        class B(Generic[T]):
            ...

        assert solve_typevar(B[A], T) == A

    def test_solve_typevar_deep(self) -> None:
        T = TypeVar("T")
        U = TypeVar("U")
        V = TypeVar("V")

        class A:
            pass

        class B:
            pass

        class C(Generic[T, U]):
            pass

        class D(C[A, B]):
            pass

        class E(D):
            pass

        class F(E, Generic[V]):
            pass

        class G:
            pass

        class H(F[G]):
            pass

        class X(H):
            pass

        # resolve T
        assert solve_typevar(D, T) == A
        assert solve_typevar(E, T) == A
        assert solve_typevar(F, T) == A
        assert solve_typevar(H, T) == A
        # resolve U
        assert solve_typevar(D, U) == B
        assert solve_typevar(E, U) == B
        assert solve_typevar(F, U) == B
        assert solve_typevar(H, U) == B
        # resilve V
        assert solve_typevar(H, V) == G

    def test_solve_typevar_chain(self) -> None:
        T = TypeVar("T")
        U = TypeVar("U")
        V = TypeVar("V")

        # typevar chain of different typevars

        class A:
            pass

        class B(Generic[T]):
            pass

        class C(B[U]):
            pass

        class D(C[V]):
            pass

        class E(D[A]):
            pass

        assert solve_typevar(E, T) == A
        assert solve_typevar(E, U) == A
        assert solve_typevar(E, V) == A
